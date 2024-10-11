import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernelSize=3,
        stride=1,
        dilation=1,
        bias=False,
        BatchNorm=None,
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwiseConv = nn.Conv2d(
            in_channels,
            in_channels,
            kernelSize,
            stride,
            0,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.BatchNorm = BatchNorm(in_channels)
        self.pointwiseConv = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
        )

    def forward(self, x):
        x = self.getPadded(
            x, self.depthwiseConv.kernel_size[0], self.depthwiseConv.dilation[0]
        )
        x = self.depthwiseConv(x)
        x = self.BatchNorm(x)
        x = self.pointwiseConv(x)
        return x

    def getPadded(self, input, kernelSize, dilation):
        kernelSizeEffective = kernelSize + (kernelSize - 1) * (dilation - 1)
        padTotal = kernelSizeEffective - 1
        padBeg = padTotal // 2
        padEnd = padTotal - padBeg
        paddedInput = F.pad(input, [padBeg, padEnd, padBeg, padEnd])
        return paddedInput


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        reps,
        stride=1,
        dilation=1,
        BatchNorm=None,
        startingRelu=True,
        growFirst=True,
        isLast=False,
    ):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=False
            )
            self.skipBN = BatchNorm(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels
        if growFirst:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_channels, out_channels, 3, 1, dilation, BatchNorm=BatchNorm
                )
            )
            rep.append(BatchNorm(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm)
            )
            rep.append(BatchNorm(filters))

        if not growFirst:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    in_channels, out_channels, 3, 1, dilation, BatchNorm=BatchNorm
                )
            )
            rep.append(BatchNorm(out_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(out_channels, out_channels, 3, 2, BatchNorm=BatchNorm)
            )
            rep.append(BatchNorm(out_channels))

        if stride == 1 and isLast:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(out_channels, out_channels, 3, 1, BatchNorm=BatchNorm)
            )
            rep.append(BatchNorm(out_channels))

        if not startingRelu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, input):
        x = self.rep(input)

        if self.skip is not None:
            skip = self.skip(input)
            skip = self.skipBN(skip)
        else:
            skip = input

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """Modified Aligned Xception"""

    def __init__(self, in_channels, outputStride, BatchNorm, pretrained=True):
        super(AlignedXception, self).__init__()
        self.namespace = "X"

        if outputStride == 16:
            entryBlockStride = 2
            middleBlockDilation = 1
            exitBlockDilations = (1, 2)
        elif outputStride == 8:
            entryBlockStride = 1
            middleBlockDilation = 2
            exitBlockDilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(
            64, 128, reps=2, stride=2, BatchNorm=BatchNorm, startingRelu=False
        )
        self.block2 = Block(
            128,
            256,
            reps=2,
            stride=2,
            BatchNorm=BatchNorm,
            startingRelu=False,
            growFirst=True,
        )
        self.block3 = Block(
            256,
            728,
            reps=2,
            stride=entryBlockStride,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
            isLast=True,
        )

        # Middle flow
        self.block4 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block5 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block6 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block7 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block8 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block9 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block10 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block11 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block12 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block13 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block14 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block15 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block16 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block17 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block18 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )
        self.block19 = Block(
            728,
            728,
            reps=3,
            stride=1,
            dilation=middleBlockDilation,
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=True,
        )

        # Exit flow
        self.block20 = Block(
            728,
            1024,
            reps=2,
            stride=1,
            dilation=exitBlockDilations[0],
            BatchNorm=BatchNorm,
            startingRelu=True,
            growFirst=False,
            isLast=True,
        )
        self.conv3 = SeparableConv2d(
            1024, 1536, 3, stride=1, dilation=exitBlockDilations[1], BatchNorm=BatchNorm
        )
        self.bn3 = BatchNorm(1536)
        self.conv4 = SeparableConv2d(
            1536, 1536, 3, stride=1, dilation=exitBlockDilations[1], BatchNorm=BatchNorm
        )
        self.bn4 = BatchNorm(1536)
        self.conv5 = SeparableConv2d(
            1536, 2048, 3, stride=3, dilation=exitBlockDilations[1], BatchNorm=BatchNorm
        )
        self.bn5 = BatchNorm(2048)

        # Init weights
        self.initWeight()

        # Load pretrained model
        if pretrained:
            self.loadModel()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.relu(x)
        lowlevelFeat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, lowlevelFeat

    def initWeight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def loadModel(self):
        pretrainDict = model_zoo.load_url(
            "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"
        )
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrainDict.items():
            if k in model_dict:
                if "pointwise" in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith("block11"):
                    model_dict[k] = v
                    model_dict[k.replace("block11", "block12")] = v
                    model_dict[k.replace("block11", "block13")] = v
                    model_dict[k.replace("block11", "block14")] = v
                    model_dict[k.replace("block11", "block15")] = v
                    model_dict[k.replace("block11", "block16")] = v
                    model_dict[k.replace("block11", "block17")] = v
                    model_dict[k.replace("block11", "block18")] = v
                    model_dict[k.replace("block11", "block19")] = v
                elif k.startswith("block12"):
                    model_dict[k.replace("block12", "block20")] = v
                elif k.startswith("bn3"):
                    model_dict[k] = v
                    model_dict[k.replace("bn3", "bn4")] = v
                elif k.startswith("conv4"):
                    model_dict[k.replace("conv4", "conv5")] = v
                elif k.startswith("bn4"):
                    model_dict[k.replace("bn4", "bn5")] = v
                else:
                    model_dict[k] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
