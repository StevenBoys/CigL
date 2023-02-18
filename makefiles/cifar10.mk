## cifar10.ERK.%: Run CIFAR 10 ERK runs. % in RigL, SNFS, SET
cifar10.ERK.%:
	${PYTHON} main.py dataset=CIFAR10 optimizer=SGD masking=$* \
	+specific=cifar10_resnet50_masking seed=$(SEED) exp_name="$*_ERK" \
	masking.density=$(DENSITY) masking.dpnum=$(DPNUM) wandb.use=$(USE_WANDB) $(HYDRA_FLAGS)

CIFAR10_DEPS := cifar10.ERK.RigL 

## cifar10: Main CIFAR 10 Runs
cifar10: $(CIFAR10_DEPS)