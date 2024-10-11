import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
        nlin_layer=nn.ReLU,
    ):
        super(ConvolutionalBatchNorm, self).__init__()
        layers = []
        if stride == 1:
            layers.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
        else:
            layers.append(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            )
        layers.append(norm_layer(out_channels))
        layers.append(nlin_layer(inplace=True))
        self.convBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.convBN(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(),
        )

    def forward(self, x):
        batches, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batches, channels)
        y = self.fc(y).view(batches, channels, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def makeDivisible(x, divisible_by=8):
    return int(np.ceil(x * 1 / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel, exp, se=False, nl="RE", stride=1
    ):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.residualConnect = stride == 1 and in_channels == out_channels

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == "RE":
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == "HS":
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(in_channels, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        if self.residualConnect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(
        self,
        in_channels=3,
        output_stride=16,
        batchNorm=nn.BatchNorm2d,
        mode="large",
        width_mult=1,
    ):
        super(MobileNetV3, self).__init__()
        self.namespace = "M3"
        firstChannel = 16
        lastChannel = 1280
        if mode == "large":
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "RE", 1],
                [3, 64, 24, False, "RE", 2],
                [3, 72, 24, False, "RE", 1],
                [5, 72, 40, True, "RE", 2],
                [5, 120, 40, True, "RE", 1],
                [5, 120, 40, True, "RE", 1],
                [3, 240, 80, False, "HS", 2],
                [3, 200, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 480, 112, True, "HS", 1],
                [3, 672, 112, True, "HS", 1],
                [5, 672, 160, True, "HS", 2],
                [5, 960, 160, True, "HS", 1],
                [5, 960, 320, True, "HS", 1],
            ]
        elif mode == "small":
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "RE", 2],
                [3, 72, 24, False, "RE", 2],
                [3, 88, 24, False, "RE", 1],
                [5, 96, 40, True, "HS", 2],
                [5, 240, 40, True, "HS", 1],
                [5, 240, 40, True, "HS", 1],
                [5, 120, 48, True, "HS", 1],
                [5, 144, 48, True, "HS", 1],
                [5, 288, 96, True, "HS", 2],
                [5, 576, 96, True, "HS", 1],
                [5, 576, 96, True, "HS", 1],
            ]
        else:
            raise NotImplementedError

        # Build first layer:
        lastChannel = (
            makeDivisible(lastChannel * width_mult) if width_mult > 1.0 else lastChannel
        )
        layers = [
            ConvolutionalBatchNorm(in_channels, firstChannel, 2, nlin_layer=Hswish)
        ]
        currentStride = 2

        # Build mobile blocks:
        for kernel, exp, channel, se, nl, s in mobile_setting:
            if currentStride == output_stride:
                stride = 1
            else:
                currentStride *= s
                stride = s
            outputChannel = makeDivisible(channel * width_mult)
            expChannel = makeDivisible(exp * width_mult)
            layers.append(
                MobileBottleneck(
                    firstChannel, outputChannel, kernel, expChannel, se, nl, stride
                )
            )
            firstChannel = outputChannel

        # Build last several layers: (disabled for DeepLabV3)
        # if mode == 'large':
        #     lastConv = makeDivisible(960 * width_mult)
        #     layers.append(ConvolutionalBatchNorm(firstChannel, lastConv, nlin_layer=Hswish))
        #     highFeatLayers.append(nn.AdaptiveAvgPool2d(1))
        #     layers.append(nn.Conv2d(lastConv, lastChannel, 1, 1, 0))
        #     layers.append(Hswish(in_channelslace=True))
        # elif mode == 'small':
        #     lastConv = makeDivisible(576 * width_mult)
        #     layers.append(ConvolutionalBatchNorm(firstChannel, lastConv, nlin_layer=Hswish))
        #     layers.append(nn.AdaptiveAvgPool2d(1))
        #     layers.append(nn.Conv2d(lastConv, lastChannel, 1, 1, 0))
        #     layers.append(Hswish(in_channelslace=True))
        # else:
        #     raise NotImplementedError

        # Build feature layers:
        layers = nn.Sequential(*layers)
        self.lowFeatLayers = layers[0:4]
        self.highFeatLayers = layers[4:]

        self.initWeights()

    def forward(self, x):
        lowLevelFeatures = self.lowFeatLayers(x)
        x = self.highFeatLayers(lowLevelFeatures)
        # x = x.mean(3).mean(2)
        # x = self.classifierLayers(x)
        return x, lowLevelFeatures

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# if __name__ == '__main__':
#     import time
#     model = MobileNetV3()
#     image = torch.randn([1,3,1920,1208])
#     startTime = time.clock()
#     output, lowlevelfeat = model(image)
#     endTime = time.clock()
#     print(lowlevelfeat.size())
#     print(output.size())
#     print(endTime-startTime)
