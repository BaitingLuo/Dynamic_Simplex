import torch
import torch.nn as nn
import torch.nn.functional as F

from .Backbone.Xception import AlignedXception
from .Backbone.Resnet import ResNet
from .Backbone.MobileNetV2 import MobileNetV2
from .Backbone.MobileNetV3 import MobileNetV3
from .Module.ASPP import ASPP
from .Module.Decoder import Decoder
import numpy as np
import matplotlib.pyplot as plt


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels,
        nClasses,
        backboneModel,
        outputStride=16,
        syncBN=False,
        freezeBN=False,
    ):
        super(DeepLabV3Plus, self).__init__()

        self.in_channels = in_channels
        self.numClasses = nClasses
        if syncBN:
            batchNorm = nn.SyncBatchNorm
        else:
            batchNorm = nn.BatchNorm2d

        self.backboneModel = backboneModel
        if backboneModel == "xception":
            self.backbone = AlignedXception(in_channels, outputStride, batchNorm)
        elif backboneModel == "resnet":
            self.backbone = ResNet(in_channels, outputStride, batchNorm)
        elif backboneModel == "mobilenetv2":
            self.backbone = MobileNetV2(in_channels, outputStride, batchNorm)
        elif backboneModel == "mobilenetv3":
            self.backbone = MobileNetV3(in_channels, outputStride, batchNorm)
        else:
            NotImplementedError

        self.aspp = ASPP(backboneModel, outputStride, batchNorm)
        self.decoder = Decoder(nClasses, backboneModel, batchNorm)

    def forward(self, input):
        x, lowlevelFeat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, lowlevelFeat)
        # temp = lowlevelFeat.cpu().detach().numpy()
        # img = np.transpose(temp, (2, 3, 1, 0))
        # fig = plt.figure()
        # for i in range(2):
        #     # for j in range(4):
        #     a = fig.add_subplot(1, 2, i + 1)
        #     imgplot = plt.imshow(img[:, :, 20, i])
        # fig.suptitle('output of convlstm, batchsize=2, channel_id=20, last time step', fontsize=16)
        # # print(img[:, :, 0, 0])
        # plt.show()
        # plt.close()
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
        return x

    def fetchModelParams(self, model):
        if model == "backbone":
            modules = self.backbone
        elif model == "aspp":
            modules = self.aspp
        elif model == "decoder":
            modules = self.decoder
        else:
            NotImplementedError

        for m in modules.named_modules():
            if (
                isinstance(m[1], nn.Conv2d)
                or isinstance(m[1], nn.SyncBatchNorm)
                or isinstance(m[1], nn.BatchNorm2d)
            ):
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

    def getParams(self, lr, decoder_only=False):
        if decoder_only:
            params = [{"params": self.fetchModelParams("decoder"), "lr": 10 * lr}]

        else:
            params = [
                {"params": self.fetchModelParams("backbone"), "lr": lr},
                {"params": self.fetchModelParams("aspp"), "lr": 10 * lr},
                {"params": self.fetchModelParams("decoder"), "lr": 10 * lr},
            ]
        return params
