from models.wide_resnet import WideResNet
from models.resnet import ResNet, BasicBlock, BottleNeck

registry = {
    #"resnet50": (ResNet, [BottleNeck, [3, 4, 6, 3], 100]),
    "resnet50": (ResNet, [BottleNeck, [3, 4, 6, 3], 100]),
    #"resnet50": (ResNet, [BasicBlock, [3, 4, 6, 3], 100]),
    "wrn-22-2": (WideResNet, [22, 2, 100, 0.0]),
}
