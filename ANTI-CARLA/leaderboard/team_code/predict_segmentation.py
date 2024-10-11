import yaml
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from Source.Utils.BaseModelHandler import BaseModelHandler
from Source.Utils.ConfigParser import BaseConfigParser

def init_model():
    with open("Config/dataset.yaml", "r") as thisFile:
        dataset = yaml.load(thisFile, Loader=yaml.BaseLoader)
    with open("Config/model.yaml", "r") as thisFile:
        model = yaml.load(thisFile, Loader=yaml.BaseLoader)
    with open("Config/optimizer.yaml", "r") as thisFile:
        optimizer = yaml.load(thisFile, Loader=yaml.BaseLoader)

    datasets = list(dataset.keys())
    backbones = list(model["DeepLabV3"]["Backbone"].keys())
    optimizers = list(optimizer.keys())

    parser = argparse.ArgumentParser(description="Semantic Segmentation on DeepLabV3+")
    # Required arguments:
    parser.add_argument("--dataset", choices=datasets, default='CARLA', help="Choose dataset")
    parser.add_argument("--loc", default='./data', help="Specify dataset location")

    # Optional arguments:
    parser.add_argument(
        "--backbone", choices=backbones, default=backbones[0], help="Choose backbone"
    )
    parser.add_argument(
        "--optimizer", choices=optimizers, default=optimizers[0], help="Choose optimizer"
    )

    args = parser.parse_args()

    with open("Config/{}.json".format('CARLA'), "r") as thisFile:
        dataset_json = json.load(thisFile)
        dataset["mean"] = dataset_json["mean"]
        dataset["std"] = dataset_json["std"]
        dataset["class"] = dataset_json["class"]

    config = BaseConfigParser(args, dataset, model, optimizer)
    model = BaseModelHandler(config)

    mean = dataset["mean"]
    std = dataset["std"]

    return model, mean, std

def get_semantic_score(model,image,mean,std):
    image = np.array(image) / 255
    image = (image - mean) / std

    image = image.transpose([2, 0, 1])
    image = torch.from_numpy(image).float()
    image = torch.unsqueeze(image,0)

    softmax_map = model.getEvaluation(image)

    return np.mean(softmax_map), softmax_map

if __name__ == "__main__":

    model, mean, std = init_model()

    with open('./data/0002.png', "rb") as thisFile:
            image = Image.open(thisFile).convert("RGB")

    softmax_score, softmax_map = get_semantic_score(model,image,mean,std)

    print(softmax_score)

    soft_image = Image.fromarray(255*softmax_map).convert("P")
    soft_image.save('./data/softmax_0002.png')
