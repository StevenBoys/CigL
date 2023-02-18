import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt
import copy
import pickle

from data import get_dataloaders
from loss import LabelSmoothingCrossEntropy
from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from sparselearning.utils.smoothen_value import SmoothenValue
from sparselearning.utils import layer_wise_density
from sparselearning.utils.train_helper import (
    get_optimizer,
    load_weights,
    save_weights,
)
from sparselearning.utils.utils_ens import (
    moving_average,
    bn_update,
)
if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def train(
    model: "nn.Module",
    mask: "Masking",
    train_loader: "DataLoader",
    optimizer: "optim",
    lr_scheduler: "lr_scheduler",
    global_step: int,
    epoch: int,
    device: torch.device,
    label_smoothing: float = 0.0,
    log_interval: int = 100,
    use_wandb: bool = False,
    masking_apply_when: str = "epoch_end",
    masking_interval: int = 1,
    masking_end_when: int = -1,
    masking_print_FLOPs: bool = False,
) -> "Union[float,int]":
    assert masking_apply_when in ["step_end", "epoch_end"]
    model.train()
    _mask_update_counter = 0
    _loss_collector = SmoothenValue()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(label_smoothing)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = smooth_CE(output, target)
        loss.backward()
        _loss_collector.add_value(loss.item())

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if (
            mask
            and masking_apply_when == "step_end"
            and global_step < masking_end_when
            and ((global_step + 1) % masking_interval) == 0
        ):
            mask.update_connections()
            _mask_update_counter += 1
        else:
            stepper.step()

        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)
        global_step += 1

        if batch_idx % log_interval == 0:
            msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f}"
            pbar.set_description(msg)

            if use_wandb:
                log_dict = {"train_loss": loss, "lr": lr_scheduler.get_lr()[0]}
                if mask:
                    density = mask.stats.total_density
                    log_dict = {
                        **log_dict,
                        "prune_rate": mask.prune_rate,
                        "density": density,
                    }
                wandb.log(
                    log_dict,
                    step=global_step,
                )

    density = mask.stats.total_density if mask else 1.0
    msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f} Prune Rate {mask.prune_rate if mask else 0:.5f} Density {density:.5f}"

    if masking_print_FLOPs:
        log_dict = {
            "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
            "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
        }

        log_dict_str = " ".join([f"{k}: {v:.4f}" for (k, v) in log_dict.items()])
        msg = f"{msg} {log_dict_str}"
        if use_wandb:
            wandb.log(
                {
                    **log_dict,
                    "layer-wise-density": layer_wise_density.wandb_bar(mask),
                },
                step=global_step,
            )

    logging.info(msg)

    return _loss_collector.smooth, global_step


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    cfg: 'DictConfig',
    global_step: int,
    epoch: int,
    device: torch.device,
    is_test_set: bool = False,
    use_wandb: bool = False,
    is_train_set: bool = False,
) -> "Union[float, float]":
    model.eval()

    loss = 0
    n = 0
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(0.0)  # No smoothing for val

    top_1_accuracy_ll = []
    top_5_accuracy_ll = []

    with torch.no_grad():
        idx = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            probs = np.exp(output.cpu().numpy())
            probs_max = np.reshape(np.max(probs, axis=1), (-1,1))
            preds = np.reshape([np.argmax(probs[m, ]) for m in range(probs.shape[0])], (-1,1))

            if idx == 0:
                probs_list = probs
                probs_max_list = probs_max
                preds_list = preds
                y_true = target.cpu().numpy()
            else:
                probs_list = np.concatenate((probs_list, probs), axis=0)
                probs_max_list = np.concatenate((probs_max_list, probs_max), axis=0)
                preds_list = np.concatenate((preds_list, preds), axis=0)
                y_true = np.concatenate((y_true, target.cpu().numpy()), axis=0)

            loss += smooth_CE(output, target).item()  # sum up batch loss

            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                output, target, topk=(1, 5)
            )
            top_1_accuracy_ll.append(top_1_accuracy)
            top_5_accuracy_ll.append(top_5_accuracy)

            pbar.update(1)

            idx += 1

        loss /= len(loader)
        top_1_accuracy = torch.tensor(top_1_accuracy_ll).mean()
        top_5_accuracy = torch.tensor(top_5_accuracy_ll).mean()

    val_or_test = "val" if not is_test_set else "test"
    val_or_test = val_or_test if not is_train_set else 'train'
    msg = f"{val_or_test.capitalize()} Epoch {epoch} Iters {global_step} {val_or_test} loss {loss:.6f} top-1 accuracy {top_1_accuracy:.4f} top-5 accuracy {top_5_accuracy:.4f} density level {cfg.masking.density}"
    pbar.set_description(msg)
    logging.info(msg)

    interval = 0.1
    bins = np.arange(0, 1+0.001, interval)
    idx_list =[[]]
    for idx in range(len(bins)-2):
        idx_list.append([])
    for idx in range(probs_max_list.shape[0]):
        g = int(probs_max_list[idx] * 100 // 10)
        if g > 9:
            g = 9
        idx_list[g].append(idx)

    acc_list = []; conf_list = []; size_list = []
    for i in range(len(idx_list)):
        if len(idx_list[i]) == 0:
            acc = 0
            conf = 0
        else:
            acc = sum(y_true[idx_list[i]] == np.reshape(preds_list[idx_list[i]], (-1))) / len(idx_list[i])
            conf = np.mean(probs_max_list[idx_list[i]])

        acc_list.append(acc)
        conf_list.append(conf)
        size_list.append(len(idx_list[i]))            

    n = sum(size_list)
    ece = 0
    for i in range(len(acc_list)):
        ece += abs(acc_list[i] - conf_list[i]) * size_list[i] / n
    print('density level ' + str(cfg.masking.density) + ': ' + str(ece))

    acc_list_final = []; conf_list_final = []
    for i in range(len(acc_list)):
        if acc_list[i] != 0 or conf_list[i] != 0:
            acc_list_final.append(acc_list[i]) 
            conf_list_final.append(conf_list[i]) 

    # Log loss, accuracy
    if use_wandb:
        wandb.log({f"{val_or_test}_loss": loss}, step=global_step)
        wandb.log({f"{val_or_test}_accuracy": top_1_accuracy}, step=global_step)
        wandb.log({f"{val_or_test}_top_5_accuracy": top_5_accuracy}, step=global_step)

    return loss, top_1_accuracy, top_5_accuracy, ece


def single_seed_run(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(**cfg.dataset)

    # Select model
    assert (
        cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0
    model = model_class(*model_args, _small_density).to(device)

    # wandb
    if cfg.wandb.use:
        with open(cfg.wandb.api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
            os.environ["WANDB_START_METHOD"] = "thread"

        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            save_code=True,
        )
        wandb.watch(model)

    # Training multiplier
    cfg.optimizer.decay_frequency *= cfg.optimizer.training_multiplier
    cfg.optimizer.decay_frequency = int(cfg.optimizer.decay_frequency)

    cfg.optimizer.epochs *= cfg.optimizer.training_multiplier
    cfg.optimizer.epochs = int(cfg.optimizer.epochs)

    if cfg.masking.get("end_when", None):
        cfg.masking.end_when *= cfg.optimizer.training_multiplier
        cfg.masking.end_when = int(cfg.masking.end_when)

    # Setup optimizers, lr schedulers
    optimizer, (lr_scheduler, warmup_scheduler) = get_optimizer(model, **cfg.optimizer)

    # Setup mask
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "T_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "T_max": max_iter,
                "T_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = Masking(
            optimizer,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )
        # Support for lottery mask
        lottery_mask_path = Path(cfg.masking.get("lottery_mask_path", ""))
        mask.add_module(model, lottery_mask_path)

    # Load from checkpoint
    model, optimizer, mask, step, start_epoch, best_val_loss = load_weights(
        model, optimizer, mask, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    )

    # Train model
    epoch = 0
    warmup_steps = cfg.optimizer.get("warmup_steps", 0)
    warmup_epochs = warmup_steps / len(train_loader)

    if (cfg.masking.print_FLOPs and cfg.wandb.use) and (start_epoch, step == (0, 0)):
        if mask:
            # Log initial inference flops etc
            log_dict = {
                "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
                "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
                "layer-wise-density": layer_wise_density.wandb_bar(mask),
            }
            wandb.log(log_dict, step=0)

    not_begin_swa = 1
    for epoch in range(start_epoch, cfg.optimizer.epochs):
        # step here is training iters not global steps
        _masking_args = {}
        if mask:
            _masking_args = {
                "masking_apply_when": cfg.masking.apply_when,
                "masking_interval": cfg.masking.interval,
                "masking_end_when": cfg.masking.end_when,
                "masking_print_FLOPs": cfg.masking.get("print_FLOPs", False),
            }

        scheduler = lr_scheduler if (epoch >= warmup_epochs) else warmup_scheduler
        _, step = train(
            model,
            mask,
            train_loader,
            optimizer,
            scheduler,
            step,
            epoch + 1,
            device,
            label_smoothing=cfg.optimizer.label_smoothing,
            log_interval=cfg.log_interval,
            use_wandb=cfg.wandb.use,
            **_masking_args,
        )
        begin_save = cfg.begin_save

        # check whether begin WMA
        if epoch + 1 >= begin_save:
            if not_begin_swa:
                swa_num = 1
                model_swa = copy.deepcopy(model)
                not_begin_swa = 0
            else:
                # begin WMA
                moving_average(model_swa, model, 1/(swa_num+1))
                swa_num += 1
                bn_update(train_loader, model_swa)        

        # Run validation
        if epoch % cfg.val_interval == 0:
            val_loss, val_accuracy, _, _ = evaluate(
                model,
                val_loader,
                cfg,
                step,
                epoch + 1,
                device,
                use_wandb=cfg.wandb.use,
            )

            # Save weights
            if (epoch + 1 == cfg.optimizer.epochs) or (
                (epoch + 1) % cfg.ckpt_interval == 0
            ):
                if val_loss < best_val_loss:
                    is_min = True
                    best_val_loss = val_loss
                else:
                    is_min = False

                save_weights(
                    model,
                    optimizer,
                    mask,
                    val_loss,
                    step,
                    epoch + 1,
                    ckpt_dir=cfg.ckpt_dir,
                    is_min=is_min,
                )

        # Apply mask
        if (
            mask
            and cfg.masking.apply_when == "epoch_end"
            and epoch < cfg.masking.end_when
        ):
            if epoch % cfg.masking.interval == 0:
                mask.restore_w_raw()
                mask.update_connections()
                mask.begin_mc_dp()

    # evaluate the produced sparse model from CigL
    loss, top_1_accuracy, top_5_accuracy, ece = evaluate(
        model_swa,
        test_loader,
        cfg,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
    )

    loss_tr, top_1_accuracy_tr, top_5_accuracy_tr, ece_tr = evaluate(
        model_swa,
        train_loader,
        cfg,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
        is_train_set=True,
    )
    print('testing: loss: {}, top_1_accuracy: {}, top_5_accuracy: {}, ece: {}'.format(loss, top_1_accuracy.item(), top_5_accuracy.item(), ece))
    print('training: loss: {}, top_1_accuracy: {}, top_5_accuracy: {}, ece: {}'.format(loss_tr, top_1_accuracy_tr.item(), top_5_accuracy_tr.item(), ece_tr))

    # save the produced sparse model from CigL
    save_weights(
        model_swa,
        optimizer,
        mask,
        val_loss,
        step,
        999,
        ckpt_dir=cfg.ckpt_dir,
        is_min=is_min,
    )

    if cfg.wandb.use:
        # Close wandb context
        wandb.join()

    return val_accuracy


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig) -> float:
    if cfg.multi_seed:
        val_accuracy_ll = []
        for seed in cfg.multi_seed:
            run_cfg = deepcopy(cfg)
            run_cfg.seed = seed
            run_cfg.ckpt_dir = f"{cfg.ckpt_dir}_seed={seed}"
            val_accuracy = single_seed_run(run_cfg)
            val_accuracy_ll.append(val_accuracy)

        return sum(val_accuracy_ll) / len(val_accuracy_ll)
    else:
        val_accuracy = single_seed_run(cfg)
        return val_accuracy


if __name__ == "__main__":
    main()
