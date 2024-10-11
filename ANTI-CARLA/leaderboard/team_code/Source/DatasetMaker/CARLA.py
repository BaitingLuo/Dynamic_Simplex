import os
import json
import numpy as np
from PIL import Image
from collections import namedtuple
from tqdm import tqdm

Label = namedtuple(
    "Label",
    [
        "name",
        "id",
        "trainId",
        "category",
        "categoryId",
        "hasInstances",
        "ignoreInEval",
        "color",
    ],
)

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label("traffic sign", 12, 0, "object", 3, False, False, (220, 220, 0)),
    Label("building", 1, 1, "construction", 2, False, False, (70, 70, 70)),
    Label("fence", 2, 2, "construction", 2, False, False, (190, 153, 153)),
    Label("other", 3, 3, "construction", 2, False, False, (250, 170, 160)),
    Label("pedestrian", 4, 4, "human", 6, True, False, (220, 20, 60)),
    Label("pole", 5, 5, "object", 3, False, False, (153, 153, 153)),
    Label("road line", 6, 6, "flat", 1, False, True, (157, 234, 50)),
    Label("road", 7, 7, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 8, "flat", 1, False, False, (244, 35, 232)),
    Label("vegetation", 9, 9, "nature", 4, False, False, (107, 142, 35)),
    Label("car", 10, 10, "vehicle", 7, True, False, (0, 0, 142)),
    Label("wall", 11, 11, "construction", 2, False, False, (102, 102, 156)),
    Label("unlabeled", 0, 12, "void", 0, False, True, (0, 0, 0)),
]

trainid2classname = {label.trainId: label.name for label in labels}
trainid2color = {label.trainId: list(label.color) for label in labels}

trainid2classname[12] = "void"
trainid2color[12] = [0, 0, 0]

classDict = {}
for i in range(13):
    classDict[i] = {"name": trainid2classname[i], "color": trainid2color[i]}


class CARLAmaker(object):
    def __init__(self, base_path, target_path):
        assert base_path.split("/")[-1] == "carla"
        sourceFolder = {}
        sourceFolder["trainSemantic"] = {
            "image": os.path.join(base_path, "seg", "images", "trainSemantic"),
            "label_color": os.path.join(
                base_path, "seg", "labels_color", "trainSemantic"
            ),
            "label": os.path.join(base_path, "seg", "labels", "trainSemantic"),
        }
        sourceFolder["trainIntrospection"] = {
            "image": os.path.join(base_path, "seg", "images", "trainIntrospection"),
            "label_color": os.path.join(
                base_path, "seg", "labels_color", "trainIntrospection"
            ),
            "label": os.path.join(base_path, "seg", "labels", "trainIntrospection"),
        }
        sourceFolder["val"] = {
            "image": os.path.join(base_path, "seg", "images", "val"),
            "label_color": os.path.join(base_path, "seg", "labels_color", "val"),
            "label": os.path.join(base_path, "seg", "labels", "val"),
        }
        sourceFolder["test"] = {
            "image": os.path.join(base_path, "seg", "images", "test"),
            "label_color": os.path.join(base_path, "seg", "labels_color", "test"),
            "label": os.path.join(base_path, "seg", "labels", "test"),
        }

        targetRoot = base_path.replace("carla", "CARLA")
        targetFolder = {}
        targetFolder["trainSemantic"] = {
            "image": os.path.join(targetRoot, target_path["base"]["image"]),
            "label": os.path.join(targetRoot, target_path["base"]["label"]),
        }
        targetFolder["trainIntrospection"] = {
            "image": os.path.join(targetRoot, target_path["introspection"]["image"]),
            "label": os.path.join(targetRoot, target_path["introspection"]["label"]),
        }
        targetFolder["val"] = {
            "image": os.path.join(targetRoot, target_path["val"]["image"]),
            "label": os.path.join(targetRoot, target_path["val"]["label"]),
        }
        targetFolder["test"] = {
            "image": os.path.join(targetRoot, target_path["test"]["image"]),
            "label": os.path.join(targetRoot, target_path["test"]["label"]),
        }

        try:
            os.mkdir(targetRoot)
            for subset in targetFolder.keys():
                os.mkdir(os.path.join(targetRoot, subset))
                for subsubset in targetFolder[subset].keys():
                    os.mkdir(targetFolder[subset][subsubset])
        except:
            raise NameError("Remove the duplicated folder at {}.".format(targetRoot))

        self.means, self.stds = [], []

        sourceSamples = self.getSamplePath(sourceFolder)
        self.pbar = tqdm(
            total=len(sourceSamples["trainSemantic"]["image"])
            + len(sourceSamples["trainIntrospection"]["image"])
            + len(sourceSamples["val"]["image"])
            + len(sourceSamples["test"]["image"])
        )
        self.pbar.set_description("Preparing dataset")

        self.copySamples(
            sourceSamples["trainSemantic"],
            sourceFolder["trainSemantic"],
            targetFolder["trainSemantic"],
        )
        self.copySamples(
            sourceSamples["trainIntrospection"],
            sourceFolder["trainIntrospection"],
            targetFolder["trainIntrospection"],
        )
        self.copySamples(sourceSamples["val"], sourceFolder["val"], targetFolder["val"])
        self.copySamples(
            sourceSamples["test"], sourceFolder["test"], targetFolder["test"]
        )

        self.pbar.close()

        imageMean = np.mean(self.means, 0).astype(np.float)
        imageSTD = np.mean(self.stds, 0).astype(np.float)

        CARLA = {"mean": list(imageMean), "std": list(imageSTD), "class": classDict}
        with open("Config/CARLA.json", "w") as thisFile:
            json.dump(CARLA, thisFile)

    def copySamples(self, sourceSample, sourceFolder, targetFolder):
        num_ids = 13
        train_colors = np.zeros((num_ids, 3), dtype=np.uint8)

        for l in labels:
            train_colors[l.trainId] = l.color

        for index in range(len(sourceSample["image"])):
            with open(sourceSample["image"][index], "rb") as thisFile:
                image = Image.open(thisFile).convert("RGB")
            with open(sourceSample["label_color"][index], "rb") as thisFile:
                label_colored = Image.open(thisFile).convert("RGB")
                label_colored = np.array(label_colored)

                label_new = np.zeros(
                    (label_colored.shape[0], label_colored.shape[1]), dtype=np.uint8
                )
                for i in range(num_ids):
                    pixels = np.all(label_colored == train_colors[i], axis=-1)
                    label_new[pixels] = i
                label_gray = Image.fromarray(label_new).convert("L")
                label = Image.fromarray(label_new).convert("P")

            normImage = np.array(image) / 255

            mean = np.mean(normImage, (0, 1))
            std = np.std(normImage, (0, 1))
            self.means.append(mean)
            self.stds.append(std)

            image.save(os.path.join(targetFolder["image"], "{:04d}.png".format(index)))
            label.save(os.path.join(targetFolder["label"], "{:04d}.png".format(index)))
            label_gray.save(
                os.path.join(sourceFolder["label"], "{:04d}.png".format(index))
            )
            self.pbar.update()

    def getSamplePath(self, folder):
        sourceSamples = {}
        for subset in folder.keys():
            images, labels = [], []
            imagenames = os.listdir(folder[subset]["image"])
            imagenames.sort()
            for imagename in imagenames:
                if ".png" in imagename:
                    namespace = imagename.split(".")[0]
                    labelname = namespace + ".png"
                    imagePath = os.path.join(folder[subset]["image"], imagename)
                    labelPath = os.path.join(folder[subset]["label_color"], labelname)
                    images.append(imagePath)
                    labels.append(labelPath)

            sourceSamples[subset] = {"image": images, "label_color": labels}

        return sourceSamples
