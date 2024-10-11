import torch
import torch.nn as nn
import torch.nn.functional as F


class moduleASPP(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernelSize, padding, dilation, BatchNorm
    ):
        super(moduleASPP, self).__init__()
        self.atrousConv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernelSize,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.BatchNorm = BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.initWeight()

    def forward(self, x):
        x = self.atrousConv(x)
        x = self.BatchNorm(x)
        x = self.relu(x)
        return x

    def initWeight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backboneModel, outputStride, BatchNorm):
        super(ASPP, self).__init__()
        if "mobilenet" in backboneModel:
            in_channels = 320
        else:
            in_channels = 2048

        if outputStride == 16:
            dilations = [1, 6, 12, 18]
        elif outputStride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = moduleASPP(
            in_channels, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm
        )
        self.aspp2 = moduleASPP(
            in_channels,
            256,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = moduleASPP(
            in_channels,
            256,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = moduleASPP(
            in_channels,
            256,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )
        self.globalAveragePool = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Conv2d(in_channels, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.initWeight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.globalAveragePool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

    def initWeight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
