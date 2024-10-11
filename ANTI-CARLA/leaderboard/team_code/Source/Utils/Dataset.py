import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, mean, std, path, output_size=513, isIntrospective=False):
        self.mean = mean
        self.std = std
        self.outputSize = output_size
        self.images = []
        self.labels = []

        imageFolder = path["image"]
        labelFolder = path["label"]

        filenames = os.listdir(imageFolder)
        filenames.sort()

        for filename in filenames:
            if ".jpg" in filename:
                labelname = filename.split(".")[0] + ".png"
                imagePath = os.path.join(imageFolder, filename)
                if isIntrospective:
                    labelPath = os.path.join(labelFolder, "i" + labelname)
                else:
                    labelPath = os.path.join(labelFolder, labelname)
                self.images.append(imagePath)
                self.labels.append(labelPath)
            elif ".png" in filename:
                imagePath = os.path.join(imageFolder, filename)
                if isIntrospective:
                    labelPath = os.path.join(labelFolder, "i" + filename)
                else:
                    labelPath = os.path.join(labelFolder, filename)
                self.images.append(imagePath)
                self.labels.append(labelPath)

        assert len(self.images) == len(self.labels)
        print("{} training pairs are loaded.".format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with open(self.images[index], "rb") as thisFile:
            image = Image.open(thisFile).convert("RGB")
        with open(self.labels[index], "rb") as thisFile:
            label = Image.open(thisFile).convert("P")

        assert image.size == label.size

        (originalWidth, originalHeight) = image.size

        magnification = np.random.uniform(3 / 4, 1.16)
        cropHeight = np.round(magnification * self.outputSize)
        cropWidth = cropHeight

        initialX = np.random.randint(originalWidth - cropWidth)
        initialY = np.random.randint(originalHeight - cropHeight)

        image = F.crop(image, initialY, initialX, cropHeight, cropWidth)
        label = F.crop(label, initialY, initialX, cropHeight, cropWidth)

        isFlipped = np.random.random() > 0.5
        if isFlipped:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        image = F.resize(
            image, [self.outputSize, self.outputSize], interpolation=Image.BILINEAR
        )
        label = F.resize(
            label, [self.outputSize, self.outputSize], interpolation=Image.NEAREST
        )

        image = np.array(image, dtype=np.float) / 255
        image = (image - self.mean) / self.std
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float()

        label = np.array(label, dtype=np.uint8)
        label = torch.from_numpy(label).long()

        return image, label


class FullDataset(Dataset):
    def __init__(self, mean, std, path, isIntrospective=False):
        self.mean = mean
        self.std = std
        self.images = []
        self.labels = []

        imageFolder = path["image"]
        labelFolder = path["label"]

        filenames = os.listdir(imageFolder)
        filenames.sort()

        for filename in filenames:
            if ".jpg" in filename:
                namespace = filename.split(".")[0]
                labelname = namespace + ".png"
                imagePath = os.path.join(imageFolder, filename)
                if isIntrospective:
                    labelPath = os.path.join(labelFolder, "i" + labelname)
                else:
                    labelPath = os.path.join(labelFolder, labelname)
                self.images.append(imagePath)
                self.labels.append(labelPath)
            elif ".png" in filename:
                imagePath = os.path.join(imageFolder, filename)
                if isIntrospective:
                    labelPath = os.path.join(labelFolder, "i" + filename)
                else:
                    labelPath = os.path.join(labelFolder, filename)
                self.images.append(imagePath)
                self.labels.append(labelPath)

        assert len(self.images) == len(self.labels)
        print("{} full size sample pairs are loaded.".format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with open(self.images[index], "rb") as thisFile:
            image = Image.open(thisFile).convert("RGB")
        with open(self.labels[index], "rb") as thisFile:
            label = Image.open(thisFile).convert("P")

        assert image.size == label.size

        image = np.array(image, dtype=np.float) / 255
        image = (image - self.mean) / self.std
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float()

        label = np.array(label, dtype=np.uint8)
        label = torch.from_numpy(label).long()

        return image, label

    def getFilepath(self, index):
        labelPath = self.labels[index]
        filename = labelPath.split("/")[-1]
        ilabelPath = labelPath.replace(filename, "i" + filename)
        return ilabelPath

    def getIntroResultPath(self, index):
        labelPath = self.labels[index]
        filename = labelPath.split("/")[-1]
        filename_new = filename.replace("i", "f")
        failureProbPath = labelPath.replace(filename, filename_new)
        return failureProbPath


class DualDataset(Dataset):
    def __init__(self, mean, std, path):
        self.mean = mean
        self.std = std
        self.images = []
        self.labels = []
        self.ilabels = []

        imageFolder = path["image"]
        labelFolder = path["label"]

        filenames = os.listdir(imageFolder)
        filenames.sort()

        for filename in filenames:
            if ".jpg" in filename:
                labelname = filename.split(".")[0] + ".png"
                imagePath = os.path.join(imageFolder, filename)
                labelPath = os.path.join(labelFolder, labelname)
                ilabelPath = os.path.join(labelFolder, "i" + labelname)
                if os.path.isfile(ilabelPath):
                    self.images.append(imagePath)
                    self.labels.append(labelPath)
                    self.ilabels.append(ilabelPath)
                else:
                    raise NameError("Error label for {} is not found!".format(filename))

            elif ".png" in filename:
                imagePath = os.path.join(imageFolder, filename)
                labelPath = os.path.join(labelFolder, filename)
                ilabelPath = os.path.join(labelFolder, "i" + filename)
                if os.path.isfile(ilabelPath):
                    self.images.append(imagePath)
                    self.labels.append(labelPath)
                    self.ilabels.append(ilabelPath)
                else:
                    raise NameError("Error label for {} is not found!".format(filename))

        assert len(self.images) == len(self.labels)
        assert len(self.images) == len(self.ilabels)
        print("{} dual sample pairs are loaded.".format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with open(self.images[index], "rb") as thisFile:
            image = Image.open(thisFile).convert("RGB")
        with open(self.labels[index], "rb") as thisFile:
            label = Image.open(thisFile).convert("P")
        with open(self.ilabels[index], "rb") as thisFile:
            ilabel = Image.open(thisFile).convert("P")

        assert image.size == label.size
        assert image.size == ilabel.size

        image = np.array(image, dtype=np.float) / 255
        image = (image - self.mean) / self.std
        image = image.transpose([2, 0, 1])
        image = torch.from_numpy(image).float()

        label = np.array(label, dtype=np.uint8)
        label = torch.from_numpy(label).long()

        ilabel = np.array(ilabel, dtype=np.uint8)
        ilabel = torch.from_numpy(ilabel).long()

        return image, label, ilabel
