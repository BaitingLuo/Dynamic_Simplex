import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        downsample=None,
        BatchNorm=None,
    ):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = BatchNorm(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self, in_channels, output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.block = BottleNeck
        self.namespace = "R"
        numBlocks = [3, 4, 23, 3]
        self.blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.makeLayer(
            64,
            numBlocks[0],
            stride=strides[0],
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.layer2 = self.makeLayer(
            128,
            numBlocks[1],
            stride=strides[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.layer3 = self.makeLayer(
            256,
            numBlocks[2],
            stride=strides[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.layer4 = self.makeMGunit(
            512,
            numBlocks[3],
            stride=strides[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )
        self.initWeight()

        if pretrained:
            self.loadModel()

    def makeLayer(self, out_channels, numBlocks, stride=1, dilation=1, BatchNorm=None):
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(out_channels * self.block.expansion),
            )
        else:
            downsample = None
        layers = []
        layers.append(
            self.block(
                self.in_channels, out_channels, stride, dilation, downsample, BatchNorm
            )
        )
        self.in_channels = out_channels * self.block.expansion
        for i in range(1, numBlocks):
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    dilation=dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    def makeMGunit(self, out_channels, numBlocks, stride=1, dilation=1, BatchNorm=None):
        if stride != 1 or self.in_channels != (out_channels * self.block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(out_channels * self.block.expansion),
            )

        layers = []
        layers.append(
            self.block(
                self.in_channels,
                out_channels,
                stride,
                dilation=self.blocks[0] * dilation,
                downsample=downsample,
                BatchNorm=BatchNorm,
            )
        )
        self.in_channels = out_channels * self.block.expansion
        for i in range(0, numBlocks):
            layers.append(
                self.block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    dilation=self.blocks[i] * dilation,
                    BatchNorm=BatchNorm,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        lowLevelFeat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, lowLevelFeat

    def initWeight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def loadModel(self):
        pretrained_dict = model_zoo.load_url(
            "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        )
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict:
                model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
