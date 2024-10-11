import os
import yaml


class BaseConfigParser(object):
    def __init__(self, args, dataset_yaml, model_yaml, optimizer_yaml):
        self.args = args
        self.dataset = dataset_yaml
        self.model = model_yaml
        self.optimizer = optimizer_yaml

        # Model parameters:
        self.backbone = self.args.backbone
        outputStride = int(self.model["DeepLabV3"]["OutputStride"])
        if (outputStride == 16) or (outputStride == 8):
            self.outputStride = outputStride
        else:
            raise NotImplementedError("The output stride should be given 8 or 16.")
        self.header = self.getHeader()
        self.outputClasses, self.voidClasses = self.getClasses()

        # Dataset parameters:
        self.path = self.getLocation()
        self.patchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["TrainPatchSize"]
        )
        self.trainBatchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["TrainBatchSize"]
        )
        self.evalBatchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["EvalBatchSize"]
        )

        # Optimizer parameters:
        self.learning_rate, self.momentum, self.weight_decay = self.getOptimParams()

    def getHeader(self):
        modelTag = "B"
        osTag = "16" if self.outputStride == 16 else "8"
        backboneTag = self.model["DeepLabV3"]["Backbone"][self.args.backbone][
            "Tag"
        ].upper()
        datasetTag = self.args.dataset.upper()
        header = "{}{}{}-{}".format(modelTag, osTag, backboneTag, datasetTag)
        return header

    def getClasses(self):
        classDict = self.dataset["class"]
        outputClasses, voidClasses = 0, 0
        for key in classDict.keys():
            outputClasses += 1
            if classDict[key]["name"].lower() == "void":
                voidClasses += 1
        return outputClasses, voidClasses

    def getLocation(self):
        root = self.args.loc
        baseSubset = self.dataset[self.args.dataset]["BaseTrain"]
        introSubset = self.dataset[self.args.dataset]["IntroTrain"]
        valSubset = self.dataset[self.args.dataset]["Validation"]
        testSubset = self.dataset[self.args.dataset]["Test"]

        basePath = {
            "image": os.path.join(root, baseSubset["Image"]),
            "label": os.path.join(root, baseSubset["Label"]),
        }
        introPath = {
            "image": os.path.join(root, introSubset["Image"]),
            "label": os.path.join(root, introSubset["Label"]),
        }
        valPath = {
            "image": os.path.join(root, valSubset["Image"]),
            "label": os.path.join(root, valSubset["Label"]),
        }
        testPath = {
            "image": os.path.join(root, testSubset["Image"]),
            "label": os.path.join(root, testSubset["Label"]),
        }
        totalPath = {
            "base": basePath,
            "introspection": introPath,
            "val": valPath,
            "test": testPath,
        }
        return totalPath

    def getOptimParams(self):
        learning_rate = self.optimizer[self.args.optimizer]["learning_rate"]
        momentum = self.optimizer[self.args.optimizer]["momentum"]
        weight_decay = self.optimizer[self.args.optimizer]["weight_decay"]
        return float(learning_rate), float(momentum), float(weight_decay)


class IntroConfigParser(object):
    def __init__(self, args, dataset_yaml, model_yaml, optimizer_yaml):
        self.args = args
        self.dataset = dataset_yaml
        self.model = model_yaml
        self.optimizer = optimizer_yaml

        self.type = args.type
        self.backbone = self.args.backbone
        outputStride = int(self.model["DeepLabV3"]["OutputStride"])
        if (outputStride == 16) or (outputStride == 8):
            self.outputStride = outputStride
        else:
            raise NotImplementedError
        self.baseHeader, self.introHeader = self.getHeader()
        self.outputClasses, self.voidClasses = self.getClasses()

        # Dataset parameters:
        self.path = self.getLocation()
        self.patchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["TrainPatchSize"]
        )
        self.trainBatchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["TrainBatchSize"]
        )
        self.evalBatchSize = int(
            self.model["DeepLabV3"]["Backbone"][self.args.backbone]["EvalBatchSize"]
        )

        # Optimizer parameters:
        self.learning_rate, self.momentum, self.weight_decay = self.getOptimParams()

        self.evalTarget = self.args.target

    def getHeader(self):
        modelTag = self.args.type.upper()[0]
        osTag = "16" if self.outputStride == 16 else "8"
        backboneTag = self.model["DeepLabV3"]["Backbone"][self.args.backbone][
            "Tag"
        ].upper()
        datasetTag = self.args.dataset.upper()
        if datasetTag == "CARLADIFF":
            baseHeader = "B{}{}-{}".format(osTag, backboneTag, "CARLA")
        else:
            baseHeader = "B{}{}-{}".format(osTag, backboneTag, datasetTag)
        introHeader = "{}{}{}-{}".format(modelTag, osTag, backboneTag, datasetTag)
        return baseHeader, introHeader

    def getClasses(self):
        classDict = self.dataset["class"]
        outputClasses, voidClasses = 0, 0
        for key in classDict.keys():
            outputClasses += 1
            if classDict[key]["name"].lower() == "void":
                voidClasses += 1
        return outputClasses, voidClasses

    def getLocation(self):
        root = self.args.loc
        baseSubset = self.dataset[self.args.dataset]["BaseTrain"]
        introSubset = self.dataset[self.args.dataset]["IntroTrain"]
        valSubset = self.dataset[self.args.dataset]["Validation"]
        testSubset = self.dataset[self.args.dataset]["Test"]

        basePath = {
            "image": os.path.join(root, baseSubset["Image"]),
            "label": os.path.join(root, baseSubset["Label"]),
        }
        introPath = {
            "image": os.path.join(root, introSubset["Image"]),
            "label": os.path.join(root, introSubset["Label"]),
        }
        valPath = {
            "image": os.path.join(root, valSubset["Image"]),
            "label": os.path.join(root, valSubset["Label"]),
        }
        testPath = {
            "image": os.path.join(root, testSubset["Image"]),
            "label": os.path.join(root, testSubset["Label"]),
        }
        totalPath = {
            "base": basePath,
            "introspection": introPath,
            "val": valPath,
            "test": testPath,
        }
        return totalPath

    def getOptimParams(self):
        learning_rate = self.optimizer[self.args.optimizer]["learning_rate"]
        momentum = self.optimizer[self.args.optimizer]["momentum"]
        weight_decay = self.optimizer[self.args.optimizer]["weight_decay"]
        return float(learning_rate), float(momentum), float(weight_decay)


class DatasetConfigParser(object):
    def __init__(self, args, datset_yaml):
        self.args = args
        self.dataset = datset_yaml
        self.target_path = self.getTargetLocation()

    def getTargetLocation(self):
        baseSubset = self.dataset[self.args.dataset]["BaseTrain"]
        introSubset = self.dataset[self.args.dataset]["IntroTrain"]
        valSubset = self.dataset[self.args.dataset]["Validation"]
        testSubset = self.dataset[self.args.dataset]["Test"]

        basePath = {"image": baseSubset["Image"], "label": baseSubset["Label"]}
        introPath = {"image": introSubset["Image"], "label": introSubset["Label"]}
        valPath = {"image": valSubset["Image"], "label": valSubset["Label"]}
        testPath = {"image": testSubset["Image"], "label": testSubset["Label"]}
        targetPath = {
            "base": basePath,
            "introspection": introPath,
            "val": valPath,
            "test": testPath,
        }
        return targetPath
