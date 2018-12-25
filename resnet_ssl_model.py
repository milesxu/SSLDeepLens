import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class BasicBlockV2(nn.Module):
    pass


class BottleneckV2(nn.Module):
    # TODO: whether test expansion being 4 is better!
    expansion = 2

    def __init__(self, in_channels, channels, stride=1, down_sample=None,
                 active='elu', alpha=1.0):
        super(BottleneckV2, self).__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.active = nn.ELU(alpha=alpha, inplace=True) if active == 'elu' \
            else nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, input):
        shortcut = input if self.down_sample is None \
            else self.down_sample(input)

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.active(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.active(out)

        return out


class ResNetSSL(nn.Module):
    # TODO: compute padding
    # TODO: add dropout
    def __init__(self, layers, block=BottleneckV2, input_channels=4,
                 num_classes=2, active='elu', alpha=1.0,
                 zero_init_residual=False, first_max_pool=False,
                 dropout=True):
        super(ResNetSSL, self).__init__()
        self.first_max_pool = first_max_pool
        self.dropout = dropout
        self.num_channels = 32
        self.conv1 = nn.Conv2d(
            input_channels, self.num_channels, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_layers = self._make_block_layers(block, layers)
        self.active = nn.ELU(alpha, inplace=True) if active == 'elu' \
            else nn.ReLU(inplace=True)
        # TODO: test the difference of AvgPool2d and AdaptiveAvgPool2d
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_block_layers(self, block, layers):
        layers_seq = []
        in_channels, out_channels = self.num_channels, 16
        for i, layer in enumerate(layers):
            stride, down_sample = 1 + (i > 0), None
            if stride != 1 or \
                    self.num_channels != out_channels * block.expansion:
                down_sample = nn.Sequential(
                    conv1x1(in_channels, out_channels * block.expansion,
                            stride=2),
                    nn.BatchNorm2d(out_channels * block.expansion))

            layers_seq.append(
                block(in_channels, out_channels, stride=stride,
                      down_sample=down_sample))
            in_channels = out_channels * block.expansion
            for _ in range(1, layer):
                layers_seq.append(
                    block(in_channels, out_channels))
            out_channels = out_channels * block.expansion
            if i == 0 and self.dropout:
                layers_seq.append(nn.Dropout2d())
        return nn.Sequential(*layers_seq)

    def forward(self, input):
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.active(input)

        if self.first_max_pool:
            input = self.maxpool(input)

        input = self.block_layers(input)
        input = self.avgpool(input)
        input = input.view(input.size(0), -1)

        return self.fc(input), input


if __name__ == "__main__":
    net = ResNetSSL([3, 3, 3, 3, 3])
    # print(net)
    input = torch.randn(1, 4, 101, 101)
    out, h_out = net(input)
