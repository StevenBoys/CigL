# Calibrating the Rigged Lottery: Making All Tickets Reliable

<span class="img_container center" style="display: block;">
    <p align="center">
    <img alt="RigL main image" height=350 src="./img/fig.pdf" style="display:block; margin-left: auto; margin-right: auto;" title="caption" />
    <br />
    <span class="img_caption" style="display: block; text-align: center;">Test accuracy and ECE value at different sparsities.</span>
    </p>
</span>

This repository contains official PyTorch implementation for the paper:<br>
**Calibrating the Rigged Lottery: Making All Tickets Reliable**<br>
Bowen Lei, Ruqi Zhang, Dongkuan Xu, Bani Mallick

Abstract: Although sparse training has been successfully used in various resource-limited deep learning tasks to save memory, accelerate training, and reduce inference time, the reliability of the produced sparse models remains unexplored. Previous research has shown that deep neural networks tend to be over-confident, and we find that sparse training exacerbates this problem. Therefore, calibrating the sparse models is crucial for reliable prediction and decision-making. In this paper, we propose a new sparse training method to produce sparse models with improved confidence calibration. In contrast to previous research that uses only one mask to control the sparse topology, our method utilizes two masks, including a deterministic mask and a random mask. The former efficiently searches and activates important weights by exploiting the magnitude of weights and gradients. While the latter brings better exploration and finds more appropriate weight values by random updates. Theoretically, we prove our method can be viewed as a hierarchical variational approximation of a probabilistic deep Gaussian process. Extensive experiments on multiple datasets, model architectures, and sparsities show that our method reduces ECE values by up to 47.8% and simultaneously maintains or even improves accuracy with only a slight increase in computation and storage burden.

## Requirements

* `python3.8` and `pytorch`: 1.7.0+ (GPU support preferable).
* Then, `make install`. 

## Example Code

Train ResNet-50 with CigL on CIFAR10

```
make cifar10.ERK.RigL DENSITY=0.1 DPNUM=100 LR=0.1 DECAY=0.6 BS=200 SEED=0
```

Train ResNet-50 with CigL on CIFAR100

```
make cifar100.ERK.RigL DENSITY=0.2 DPNUM=10 LR=0.1 DECAY=0.6 BS=200 SEED=0
```

* `--DENSITY`: the density level of the deterministic mask and the sparsity = 1 - density.
* `--DPNUM`: the random mask's sparsity = 1/DPNUM.
* `--LR`: the initial learning rate.
* `--DECAY`: the decay rate of the piecewise constant decay schedule.
* `--BS`: the beginning epoch of the weight & mask averaging procedure.
* `--SEED`: the random seed.

Modify `makefiles/cifar10.mk` or `makefiles/cifar10.mk` to use different model architectures and sparse training methods.

## Citation

```
@inproceedings{
lei2023calibrating,
title={Calibrating the Rigged Lottery: Making All Tickets Reliable},
author={Bowen Lei and Ruqi Zhang and Dongkuan Xu and Bani Mallick},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=KdwnGErdT6}
}
```

## Credits

We built on [[Reproducibilty Challenge] RigL](https://github.com/varun19299/rigl-reproducibility).
