import torch
import torch.nn as nn
import torch.nn.functional as F

# 4. ResNet Model Specifications - Basic Residual Block
class BasicBlock(nn.Module):
    """
    Core component for ResNet-18/34.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolution: 3x3, stride
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        # Batch Normalization (BN) layer after convolution
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution: 3x3, stride=1
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Batch Normalization (BN) layer after convolution
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip Connection (Identity Mapping)
        self.shortcut = nn.Sequential()
        # If stride is not 1 OR in_planes != planes * expansion, a projection
        # is needed for the shortcut to match dimensions.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # First conv -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv -> BN
        out = self.bn2(self.conv2(out))
        # Output + Shortcut (Identity Mapping)
        out += self.shortcut(x)
        # Final ReLU
        out = F.relu(out)
        return out

# 

# 4. ResNet Model Specifications - ResNet-18
class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Initial Layer (Customized for CIFAR-10 32x32 input)
        # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stages (Downsampling using stride=2 in the first block of stages 2, 3, and 4)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Conv -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # Residual Layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global Average Pooling (32x32 -> 1x1)
        out = F.avg_pool2d(out, 4) # 8x8 input -> 4x4, 4x4 input -> 1x1
        out = out.view(out.size(0), -1)
        # Linear/Classifier Layer
        out = self.linear(out)
        return out

# Helper function to initialize ResNet-18
def ResNet18_cifar10():
    return ResNet18(num_classes=10)

if __name__ == '__main__':
    # Test the model forward pass with dummy data
    net = ResNet18_cifar10()
    y = net(torch.randn(1, 3, 32, 32))
    print(f"ResNet-18 Output shape: {y.size()}")