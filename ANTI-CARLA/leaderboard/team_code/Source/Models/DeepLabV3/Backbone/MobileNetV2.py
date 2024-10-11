import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class ConvolutionalBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride, BatchNorm):
        super(ConvolutionalBatchNorm, self).__init__()
        self.convBN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        x = self.convBN(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, dilation, expand_ratio, BatchNorm
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hiddenDimension = round(in_channels * expand_ratio)
        self.useResidualConnect = self.stride == 1 and in_channels == out_channels
        self.kernelSize = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # DW
                nn.Conv2d(
                    hiddenDimension,
                    hiddenDimension,
                    3,
                    stride,
                    0,
                    dilation,
                    groups=hiddenDimension,
                    bias=False,
                ),
                BatchNorm(hiddenDimension),
                nn.ReLU6(inplace=True),
                # PW-linear
                nn.Conv2d(hiddenDimension, out_channels, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                # PW
                nn.Conv2d(in_channels, hiddenDimension, 1, 1, 0, 1, bias=False),
                BatchNorm(hiddenDimension),
                nn.ReLU6(inplace=True),
                # DW
                nn.Conv2d(
                    hiddenDimension,
                    hiddenDimension,
                    3,
                    stride,
                    0,
                    dilation,
                    groups=hiddenDimension,
                    bias=False,
                ),
                BatchNorm(hiddenDimension),
                nn.ReLU6(inplace=True),
                # PW-linear
                nn.Conv2d(hiddenDimension, out_channels, 1, 1, 0, 1, bias=False),
                BatchNorm(out_channels),
            )

    def forward(self, x):
        paddedX = self.getFixedPadding(x, self.kernelSize, self.dilation)
        if self.useResidualConnect:
            x = x + self.conv(paddedX)
        else:
            x = self.conv(paddedX)
        return x

    def getFixedPadding(self, input, kernelSize, dilation):
        kernelSizeEffective = kernelSize + (kernelSize - 1) * (dilation - 1)
        padTotal = kernelSizeEffective - 1
        padBegin = padTotal // 2
        padEnd = padTotal - padBegin
        paddedInput = F.pad(input, (padBegin, padEnd, padBegin, padEnd))
        return paddedInput


class MobileNetV2(nn.Module):
    def __init__(
        self,
        in_channels=3,
        output_stride=16,
        BatchNorm=nn.BatchNorm2d,
        width_multiplier=1,
        pretrained=True,
    ):
        super(MobileNetV2, self).__init__()
        self.namespace = "M2"
        inputChannel = int(32 * width_multiplier)
        currentStride = 1
        rate = 1
        InvertedResidualSetting = [
            # T    C  N  S
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First Block
        layers = [ConvolutionalBatchNorm(in_channels, inputChannel, 2, BatchNorm)]
        currentStride *= 2

        # Inverted Residual Blocks
        for t, c, n, s in InvertedResidualSetting:
            if currentStride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                currentStride *= s
            outputChannel = int(c * width_multiplier)

            for i in range(n):
                if i == 0:
                    layers.append(
                        InvertedResidual(
                            inputChannel, outputChannel, stride, dilation, t, BatchNorm
                        )
                    )
                else:
                    layers.append(
                        InvertedResidual(
                            inputChannel, outputChannel, 1, dilation, t, BatchNorm
                        )
                    )
                inputChannel = outputChannel

        layers = nn.Sequential(*layers)
        self.initWeights()

        if pretrained:
            self.loadModel()

        self.lowLayers = layers[0:4]
        self.highLayers = layers[4:]

    def forward(self, x):
        lowLevelFeatures = self.lowLayers(x)
        x = self.highLayers(lowLevelFeatures)
        return x, lowLevelFeatures

    def loadModel(self):
        pretrainedParams = model_zoo.load_url(
            "http://jeff95.me/models/mobilenet_v2-6a65762b.pth"
        )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrainedParams.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# if __name__ == '__main__':
#     import time
#     model = MobileNetV2()
#     image = torch.randn([1,3,1920,1208])
#     startTime = time.clock()
#     output, lowlevelfeat = model(image)
#     endTime = time.clock()
#     print(lowlevelfeat.size())
#     print(output.size())
#     print(endTime-startTime)
