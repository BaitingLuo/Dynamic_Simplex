import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, numClasses, backboneModel, batchNorm):
        super(Decoder, self).__init__()
        if backboneModel == "resnet":
            lowlevelChannels = 256
        elif backboneModel == "xception":
            lowlevelChannels = 128
        elif "mobilenet" in backboneModel:
            lowlevelChannels = 24
        else:
            raise NotImplementedError

        self.numClasses = numClasses

        self.conv1 = nn.Conv2d(lowlevelChannels, 48, 1, bias=False)
        self.bn1 = batchNorm(48)
        self.relu = nn.ReLU()
        self.lastConv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            batchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            batchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, numClasses, kernel_size=1, stride=1),
        )
        self.initWeight()

    def forward(self, x, lowlevelFeat):
        lowlevelFeat = self.conv1(lowlevelFeat)
        lowlevelFeat = self.bn1(lowlevelFeat)
        lowlevelFeat = self.relu(lowlevelFeat)

        x = F.interpolate(
            x, size=lowlevelFeat.size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x, lowlevelFeat], dim=1)
        x = self.lastConv(x)
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
