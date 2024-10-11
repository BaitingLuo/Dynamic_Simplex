import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..Models.DeepLabV3.DeepLabV3 import DeepLabV3Plus
from .Dataset import TrainDataset, FullDataset, DualDataset
from .Evaluator import SegmentationEvaluator
import math

class BaseModelHandler(object):
    def __init__(self, config, path):
        self.header = config.header
        self.numClasses = config.outputClasses
        self.evalClasses = config.outputClasses - config.voidClasses
        self.data_path = path

        self.model = DeepLabV3Plus(
            3, config.outputClasses, config.backbone, config.outputStride
        )

        # Number of GPUs:
        self.numGPU = torch.cuda.device_count()

        # Contruct Datasets:
        self.path = config.path
        self.mean, self.std = (
            np.array(config.dataset["mean"]),
            np.array(config.dataset["std"]),
        )
        self.classDict = config.dataset["class"]

        # Checkpoint variables:
        self.epoch = 0
        self.modelParams = None
        self.optimParams = None
        self.lossX, self.lossY = [], []
        self.evalX, self.evalY = [], []
        # Mount parameters:
        self.learning_rate, self.momentum, self.weight_decay = (
            config.learning_rate,
            config.momentum,
            config.weight_decay,
        )
        self.mountCheckpoint()

        print("Baseline model is good to go.")

    def getModelOnGPU(self):
        device = torch.device("cuda")
        model = (
            nn.DataParallel(self.model, device_ids=range(self.numGPU))
            if self.numGPU > 1
            else self.model.to(device)
        )
        return model

    def getOptimizer(self, model):
        optimizer = torch.optim.SGD(
            model.getParams(self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        if self.optimParams != None:
            optimizer.load_state_dict(self.optimParams)
        return optimizer

    def getLossFunc(self):
        return nn.CrossEntropyLoss(ignore_index=255)

    def mountCheckpoint(self):
        filenames = os.listdir(self.data_path + "Params/")
        indexList = []
        for filename in filenames:
            if self.header in filename:
                index = filename.split(".")[0][-5:]
                indexList.append(int(index))

        if len(indexList) > 0:
            target_index = max(indexList)

            savefilename = self.data_path + "Params/{}{:05d}.tar".format(self.header, target_index)
            checkpoint = torch.load(savefilename,map_location=torch.device('cpu'))
            self.epoch = checkpoint["epoch"]
            self.model.backbone.load_state_dict(checkpoint["backbone"])
            self.model.aspp.load_state_dict(checkpoint["aspp"])
            self.model.decoder.load_state_dict(checkpoint["decoder"])
            self.optimParams = checkpoint["optimizer"]
            print(
                "Baseline model checkpoint of epoch {} is successfully loaded.".format(
                    target_index
                )
            )

        else:
            print("No saved checkpoint is found.")

    def getEvaluation(self,image):
        softmax_function = nn.Softmax(dim=0)

        model = self.model

        model.eval()

        with torch.no_grad():
            output = model(image).squeeze(0)

        soft_output = np.amax(softmax_function(output).numpy().astype(np.float),axis=0)

        return soft_output
