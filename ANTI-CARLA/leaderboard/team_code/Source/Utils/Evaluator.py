import torch
import numpy as np
import sklearn.metrics as skm


class IntrospectionAccuracyEvaluator(object):
    def __init__(self, evalClasses):
        self.evalClasses = evalClasses

    def __call__(self, output, label, ilabel):
        semanticLabel = label.numpy().astype(np.uint8)
        introLabel = ilabel.numpy().astype(np.uint8)
        predLabel = torch.argmax(output, 0).numpy().astype(np.uint8)

        evalArea = semanticLabel < self.evalClasses
        evalPixels = np.sum(evalArea)
        acc = np.sum(introLabel[evalArea] == predLabel[evalArea]) / evalPixels
        return acc


class SegmentationEvaluator(object):
    def __init__(self, numClasses, evalClasses):
        self.numClasses = numClasses
        self.evalClasses = evalClasses

    def __call__(self, output, label, sigmaMap):
        gtLabel = label.numpy().astype(np.uint8)
        predLabel = torch.argmax(output, 0).numpy().astype(np.uint8)
        softmaxScore = (
            torch.softmax(output, 0).numpy().astype(np.float).transpose([1, 2, 0])
        )
        totalPixels = gtLabel.shape[0] * gtLabel.shape[1]
        (
            gtPops,
            predPops,
            confusionMatrix,
            recalls,
            precisions,
            IoUs,
            STDs,
            softmaxPRAUCs,
            sigmaPRAUCs,
            softmaxAUCs,
            sigmaAUCs,
        ) = ([], [], [], [], [], [], [], [], [], [], [])

        for index in range(self.numClasses):
            gtPop = np.sum(gtLabel == index) / totalPixels
            predPop = np.sum(predLabel == index) / totalPixels
            if index < self.evalClasses:
                confusionTerms = self.getConfusionMatrix(gtLabel, predLabel, index)
                recall = self.getRecall(gtLabel, predLabel, index)
                precision = self.getPrecision(gtLabel, predLabel, index)
                IoU = self.getIoU(gtLabel, predLabel, index)
                STD = self.getMeanSTD(gtLabel, sigmaMap, index)
                softmaxPRAUC = self.getSoftmaxPRAUC(
                    gtLabel, predLabel, softmaxScore, index
                )
                sigmaPRAUC = self.getSigmaPRAUC(gtLabel, predLabel, sigmaMap, index)
                softmaxAUC = self.getSoftmaxAUC(gtLabel, predLabel, softmaxScore, index)
                sigmaAUC = self.getSigmaAUC(gtLabel, predLabel, sigmaMap, index)
            else:
                confusionTerms = [0, 0, 0, 0]
                recall = 0
                precision = 0
                IoU = 0
                STD = 0
                softmaxPRAUC = 0
                sigmaPRAUC = 0
                softmaxAUC = 0
                sigmaAUC = 0

            gtPops.append(gtPop)
            predPops.append(predPop)
            confusionMatrix.append(confusionTerms)
            recalls.append(recall)
            precisions.append(precision)
            IoUs.append(IoU)
            STDs.append(STD)
            softmaxPRAUCs.append(softmaxPRAUC)
            sigmaPRAUCs.append(sigmaPRAUC)
            softmaxAUCs.append(softmaxAUC)
            sigmaAUCs.append(sigmaAUC)
        return (
            gtPops,
            predPops,
            confusionMatrix,
            recalls,
            precisions,
            IoUs,
            STDs,
            softmaxPRAUCs,
            sigmaPRAUCs,
            softmaxAUCs,
            sigmaAUCs,
        )

    def getConfusionMatrix(self, gtLabel, predLabel, index):
        label_true = (gtLabel == index).ravel()
        label_pred = (predLabel == index).ravel()
        TN, FP, FN, TP = skm.confusion_matrix(
            label_true, label_pred, labels=[False, True]
        ).ravel()
        confusionTerms = [TN, FP, FN, TP]
        return confusionTerms

    def getRecall(self, gtLabel, predLabel, index):
        evalArea = gtLabel == index
        evalPixels = np.sum(evalArea)
        if evalPixels > 0:
            recall = np.sum(gtLabel[evalArea] == predLabel[evalArea]) / evalPixels
        else:
            recall = np.nan
        return recall

    def getPrecision(self, gtLabel, predLabel, index):
        evalArea = predLabel == index
        evalPixels = np.sum(evalArea)
        if evalPixels > 0:
            precision = np.sum(gtLabel[evalArea] == predLabel[evalArea]) / evalPixels
        else:
            precision = np.nan
        return precision

    def getIoU(self, gtLabel, predLabel, index):
        gtArea = gtLabel == index
        predArea = predLabel == index
        intersectionArea = gtArea & predArea
        unionArea = gtArea | predArea
        intersectionPixels = np.sum(intersectionArea)
        unionPixels = np.sum(unionArea)

        if unionPixels > 0:
            IoU = intersectionPixels / unionPixels
        else:
            IoU = np.nan
        return IoU

    def getMeanSTD(self, gtLabel, sigmaMap, index):
        evalArea = gtLabel == index
        if np.sum(evalArea) > 0:
            meanSTD = np.mean(sigmaMap[evalArea])
        else:
            meanSTD = np.nan
        return meanSTD

    def getSoftmaxPRAUC(self, gtLabel, predLabel, softmaxScore, index):
        gtEvalArea = gtLabel == index
        predEvalArea = predLabel == index
        evalArea = gtEvalArea | predEvalArea  # represents TP+FP+FN
        scoreMap = softmaxScore[:, :, index]
        if np.sum(evalArea) > 0:  # TP+FN+FP>0, gt object exist and prediction exist
            indexArray = (
                gtLabel[evalArea] == predLabel[evalArea]
            ).ravel()  # represents TP
            scoreArray = scoreMap[evalArea].ravel()
            if np.sum(indexArray) == 0:  # if there is no TP but FP+FN>0, bad classifier
                AUC = 0
            else:
                if np.sum(indexArray) == len(
                    indexArray
                ):  # if TP>0 but FP+FN=0, skip this image
                    AUC = np.nan
                else:  # if both TP and FP+FN are exist
                    label_true = (gtLabel[evalArea]).ravel()  # not consider the TNs
                    label_true = np.where(
                        label_true == index, True, False
                    )  # set the pixels belong to the class as positive case
                    AUC = skm.average_precision_score(label_true, scoreArray)
        else:
            AUC = np.nan
        return AUC

    def getSigmaPRAUC(self, gtLabel, predLabel, sigmaMap, index):
        gtEvalArea = gtLabel == index
        predEvalArea = predLabel == index
        evalArea = gtEvalArea | predEvalArea
        if np.sum(evalArea) > 0:
            indexArray = (gtLabel[evalArea] == predLabel[evalArea]).ravel()
            indexArray = ~indexArray
            scoreArray = sigmaMap[evalArea].ravel() / np.max(sigmaMap)
            if np.sum(indexArray) == 0:
                AUC = 0
            else:
                if np.sum(indexArray) == len(indexArray):
                    AUC = np.nan
                else:
                    label_true = (gtLabel[evalArea]).ravel()
                    label_true = np.where(label_true == index, True, False)
                    AUC = skm.average_precision_score(label_true, scoreArray)
        else:
            AUC = np.nan
        return AUC

    def getSoftmaxAUC(self, gtLabel, predLabel, softmaxScore, index):
        # Todo: modify this function for AUC ROC
        gtEvalArea = gtLabel == index
        predEvalArea = predLabel == index
        evalArea = gtEvalArea | predEvalArea
        scoreMap = softmaxScore[:, :, index]
        if np.sum(evalArea) > 0:
            indexArray = (gtLabel[evalArea] == predLabel[evalArea]).ravel()
            scoreArray = scoreMap[evalArea].ravel()
            if np.sum(indexArray) == 0:
                AUC = 0
            else:
                if np.sum(indexArray) == len(indexArray):
                    AUC = np.nan
                else:
                    AUC = skm.roc_auc_score(indexArray, scoreArray)
        else:
            AUC = np.nan
        return AUC

    def getSigmaAUC(self, gtLabel, predLabel, sigmaMap, index):
        # Todo: modify this function for AUC ROC
        gtEvalArea = gtLabel == index
        predEvalArea = predLabel == index
        evalArea = gtEvalArea | predEvalArea
        if np.sum(evalArea) > 0:
            indexArray = (gtLabel[evalArea] == predLabel[evalArea]).ravel()
            indexArray = ~indexArray
            scoreArray = sigmaMap[evalArea].ravel() / np.max(sigmaMap)
            if np.sum(indexArray) == 0:
                AUC = 0
            else:
                if np.sum(indexArray) == len(indexArray):
                    AUC = np.nan
                else:
                    AUC = skm.roc_auc_score(indexArray, scoreArray)
        else:
            AUC = np.nan
        return AUC


class IntrospectionEvaluator(object):
    def __init__(self, numClasses, evalClasses):
        self.numClasses = numClasses
        self.evalClasses = evalClasses

    def __call__(self, output, label, ilabel):
        classLabel = label.numpy().astype(np.uint8)
        introLabel = ilabel.numpy().astype(np.uint8)
        predLabel = torch.argmax(output, 0).numpy().astype(np.uint8)
        scoreMap = torch.softmax(output, 0)[1, :, :].numpy()

        totalPixels = classLabel.shape[0] * classLabel.shape[1]

        (
            gtPops,
            confusionMatrix,
            ACCs,
            recalls,
            precisions,
            ROCAUCs,
            PRAUCs,
            curvePs,
            curveRs,
        ) = ([], [], [], [], [], [], [], [], [])
        for index in range(self.numClasses):
            gtPop = np.sum(classLabel == index) / totalPixels
            if index < self.evalClasses:
                # confusionTerms = self.getConfusionMatrix(
                #     classLabel, introLabel, predLabel, index
                # )
                # ACC = self.getACC(classLabel, introLabel, predLabel, index)
                # recall = self.getRecall(classLabel, introLabel, predLabel, index)
                # precision = self.getPrecision(classLabel, introLabel, predLabel, index)
                # ROCAUC = self.getROCAUC(
                #     classLabel, introLabel, predLabel, scoreMap, index
                # )
                PRAUC, curveP, curveR = self.getPRAUC(
                    classLabel, introLabel, predLabel, scoreMap, index
                )
            else:
                # confusionTerms = [0, 0, 0, 0]
                # ACC = 0
                # recall = 0
                # precision = 0
                # ROCAUC = 0
                PRAUC = 0
                curveP = np.array(
                    [0, 0]
                )  # should be an array, so set it all with zeros
                curveR = np.array([0, 0])

            gtPops.append(gtPop)
            # confusionMatrix.append(confusionTerms)
            # ACCs.append(ACC)
            # recalls.append(recall)
            # precisions.append(precision)
            # ROCAUCs.append(ROCAUC)
            PRAUCs.append(PRAUC)
            curvePs.append(curveP)
            curveRs.append(curveR)
        return (
            gtPops,
            confusionMatrix,
            ACCs,
            recalls,
            precisions,
            ROCAUCs,
            PRAUCs,
            curvePs,
            curveRs,
        )

    def getConfusionMatrix(self, classLabel, introLabel, predLabel, index):
        evalArea = classLabel == index
        evalPixels = np.sum(evalArea)
        label_true = (introLabel[evalArea]).ravel()
        label_pred = (predLabel[evalArea]).ravel()
        if evalPixels > 0:
            TN, FP, FN, TP = skm.confusion_matrix(
                label_true, label_pred, labels=[1, 0]
            ).ravel()
            confusionTerms = [TN, FP, FN, TP]
        else:
            confusionTerms = [np.nan, np.nan, np.nan, np.nan]
        return confusionTerms

    def getACC(self, classLabel, introLabel, predLabel, index):
        evalArea = classLabel == index
        evalPixels = np.sum(evalArea)

        if evalPixels > 0:
            ACC = np.sum(introLabel[evalArea] == predLabel[evalArea]) / evalPixels
        else:
            ACC = np.nan
        return ACC

    def getRecall(self, classLabel, introLabel, predLabel, index):
        evalArea = classLabel == index
        evalPixels = np.sum(evalArea)
        introLabel_new = np.where(introLabel == 0, 1, 0)
        predLabel_new = np.where(predLabel == 0, 1, 0)
        comparison_matrix = introLabel_new[evalArea] * predLabel_new[evalArea]
        introErrorPixels = np.sum(introLabel_new[evalArea])

        if (evalPixels > 0) & (introErrorPixels > 0):
            recall = np.sum(comparison_matrix) / introErrorPixels
        else:
            recall = np.nan
        return recall

    def getPrecision(self, classLabel, introLabel, predLabel, index):
        evalArea = classLabel == index
        evalPixels = np.sum(evalArea)
        introLabel_new = np.where(introLabel == 0, 1, 0)
        predLabel_new = np.where(predLabel == 0, 1, 0)
        comparison_matrix = introLabel_new[evalArea] * predLabel_new[evalArea]
        predErrorPixels = np.sum(predLabel_new[evalArea])

        if (evalPixels > 0) & (predErrorPixels > 0):
            precision = np.sum(comparison_matrix) / predErrorPixels
        else:
            precision = np.nan
        return precision

    def getROCAUC(self, classLabel, introLabel, predLabel, scoreMap, index):
        # Todo: modify this function
        classArea = classLabel == index
        comparison_matrix = introLabel[classArea] + predLabel[classArea]
        matrix_FPFN = np.where(comparison_matrix == 1, True, False)
        matrix_TP = np.where(comparison_matrix == 0, True, False)
        matrix_TN = np.where(comparison_matrix == 2, False, True)
        evalPixels = np.sum(matrix_FPFN[matrix_TN]) + np.sum(matrix_TP[matrix_TN])
        if evalPixels > 0:
            if np.sum(matrix_TP[matrix_TN]) == 0:
                AUC = 0
            else:
                if np.sum(matrix_FPFN[matrix_TN]) == 0:
                    AUC = np.nan
                else:
                    indexArray = introLabel[classArea]
                    indexArray = (indexArray[matrix_TN]).ravel()
                    if np.sum(indexArray) > 0:
                        # indexArray = (matrix_TP[matrix_TN]).ravel()
                        indexArray = np.where(indexArray == 0, 1, 0)
                        scoreArray = 1 - scoreMap[classArea]
                        scoreArray = (scoreArray[matrix_TN]).ravel()
                        AUC = skm.roc_auc_score(indexArray, scoreArray)
                    else:
                        AUC = np.nan
        else:
            AUC = np.nan
        return AUC

    def getPRAUC(self, classLabel, introLabel, predLabel, scoreMap, index):
        """
        This function can not only compute the AUC PR values for a class directly, but also return the precisions
        and recalls array for plotting PR curves later.

        Returns:
            AUC: AUC PR value for a class, using function skm.average_precision_score() directly
            curveP: an array contains precisions, using function skm.precision_recall_curve()
            curveR: an array contains recalls, using function skm.precision_recall_curve()
        """
        classArea = (
            classLabel == index
        )  # only consider the pixels belong to the class "index"
        comparison_matrix = introLabel[classArea] + predLabel[classArea]
        matrix_FPFN = np.where(comparison_matrix == 1, True, False)
        matrix_TP = np.where(comparison_matrix == 0, True, False)
        matrix_TN = np.where(
            comparison_matrix == 2, False, True
        )  # not consider the TN cases
        evalPixels = np.sum(matrix_FPFN[matrix_TN]) + np.sum(
            matrix_TP[matrix_TN]
        )  # TP+FN+FP
        if evalPixels > 0:  # TP+FN+FP>0, gt object exist and prediction exist
            if np.sum(matrix_TP[matrix_TN]) == 0:  # no TP but FP+FN>0, bad classifier
                AUC = 0
                curveP, curveR = (
                    [0, 0],
                    [0, 0],
                )  # meaning: in this case precisions and recalls are zeros.
            else:
                if (
                    np.sum(matrix_FPFN[matrix_TN]) == 0
                ):  # TP>0 but FP+FN=0, skip this image
                    AUC = np.nan
                    curveP, curveR = [np.nan, np.nan], [np.nan, np.nan]
                else:  # both TP and FP+FN exist
                    indexArray = introLabel[classArea]
                    indexArray = (indexArray[matrix_TN]).ravel()  # except the TNs
                    indexArray = np.where(
                        indexArray == 0, True, False
                    )  # set the "failure" as positive
                    scoreArray = (
                        1 - scoreMap[classArea]
                    )  # when the score around 1: "very likely a failure"
                    scoreArray = (scoreArray[matrix_TN]).ravel()
                    AUC = skm.average_precision_score(indexArray, scoreArray)
                    curveP, curveR, _ = skm.precision_recall_curve(
                        indexArray, scoreArray
                    )
        else:
            AUC = np.nan
            curveP, curveR = [np.nan, np.nan], [np.nan, np.nan]
        return AUC, curveP, curveR
