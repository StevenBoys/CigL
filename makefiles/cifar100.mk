## cifar100.ERK.%: Run CIFAR 10 ERK runs. % in RigL, SNFS, SET
cifar100.ERK.%:
	${PYTHON} main.py hydra/launcher=basic dataset=CIFAR100 optimizer=SGD masking=$* \
	+specific=cifar100_resnet50_masking seed=$(SEED) exp_name="$*_ERK" \
	masking.density=$(DENSITY)  masking.dpnum=$(DPNUM) optimizer.lr=$(LR) \
	optimizer.decay_factor=$(DECAY) begin_save=$(BS) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

CIFAR100_DEPS := cifar100.ERK.RigL 

## cifar100: Main CIFAR 10 Runs
cifar100: $(CIFAR100_DEPS)