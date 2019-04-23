'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from PCANorm import myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, Norm, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm(planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm(planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Norm(self.expansion*planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, Norm, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm(planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = Norm(planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = Norm(self.expansion*planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Norm(self.expansion*planes, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, Norm, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Norm(64, affine=True) if Norm in [nn.BatchNorm2d, nn.InstanceNorm2d, myBatchNorm, myPCANorm, myZCANorm, myPCANorm_noRec] else Norm(32, 64)
        self.layer1 = self._make_layer(nn.BatchNorm2d, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(nn.BatchNorm2d, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(nn.BatchNorm2d, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(nn.BatchNorm2d, block, 512, num_blocks[3], stride=2)
        # self.layer1 = self._make_layer(Norm, block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(Norm, block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(Norm, block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(Norm, block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, Norm, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(Norm, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlockXnorm(nn.Module):
    expansion = 1

    def __init__(self, Norm, in_planes, planes, stride=1, h=0, w=0):
        super(BasicBlockXnorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm([planes, int(h/stride), int(w/stride)], affine=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm([planes, int(h/stride), int(w/stride)], affine=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Norm([self.expansion*planes, int(h/stride), int(w/stride)], affine=True)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckXnorm(nn.Module):
    expansion = 4

    def __init__(self, Norm, in_planes, planes, stride=1, h=0, w=0):
        super(BottleneckXnorm, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm([planes, h, w])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = Norm([planes, int(h/stride), int(w/stride)])
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = Norm([self.expansion*planes, int(h/stride), int(w/stride)])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                Norm([self.expansion*planes, int(h/stride), int(w/stride)])
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetXnorm(nn.Module):
    def __init__(self, Norm, block, num_blocks, num_classes=10, h=0, w=0):
        super(ResNetXnorm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Norm([64, h, w])
        self.layer1 = self._make_layer(Norm, block, 64, num_blocks[0], stride=1, h=h, w=w)
        self.layer2 = self._make_layer(Norm, block, 128, num_blocks[1], stride=2, h=h, w=w)
        self.layer3 = self._make_layer(Norm, block, 256, num_blocks[2], stride=2, h=int(h/2), w=int(w/2))
        self.layer4 = self._make_layer(Norm, block, 512, num_blocks[3], stride=2, h=int(h/2**2), w=int(w/2**2))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, Norm, block, planes, num_blocks, stride, h, w):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(Norm, self.in_planes, planes, stride, h, w))
            [h, w] = [int(h/stride), int(w/stride)]
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Norm = nn.BatchNorm2d
# Norm = nn.GroupNorm

def ResNet18(Norm):
    return ResNet(Norm, BasicBlock, [2,2,2,2])

def ResNet34(Norm):
    return ResNet(Norm, BasicBlock, [3,4,6,3])

def ResNet50(Norm):
    if Norm is not nn.LayerNorm:
        return ResNet(Norm, Bottleneck, [3, 4, 6, 3])
    else:
        return ResNetXnorm(Norm, BottleneckXnorm, [3,4,6,3], h=32, w=32)

def ResNet101(Norm):
    return ResNet(Norm, Bottleneck, [3,4,23,3])

def ResNet152(Norm):
    return ResNet(Norm, Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
