import glob
import os
import warnings
import numpy as np
import pickle as pkl
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
from CNN_architectures import *
import matplotlib.pyplot as plt
from keras_video import VideoFrameGenerator

config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True


class AnalyzeResultsClass:
    """
    This class is used to show the results concerning the accuracy of the model on the test data, and to plot the confusion matrices and ROC and AUC curves.
    The input parameters are:

    - *modelPath*: the path where model to be tested are saved
    - *videoDatasetPath*: the dataset path that will be used for plotting the results
    - *stratifiedKFolds*: if true the models trained with KFolds strategy will be analyzed, else the models trained without the KFolds will be taken into consideration
    - *xTestPath*: the path/file containing the names of the data that form the test set. It can be a pickle file or a path of the folder with the data to be tested
    - *transformation*: the transformation applied to the original dataset. It is used to check if the model path, the video path, and the test path are consistent
    - *classesNames*: list containing the names of the classes for plotting the results
    - *CNNModel*: model on which compute the test accuracy, metrics, confusion matrices (default is model 2). Possible choices are in the range 1-9
    """

    def __init__(self,
                 modelPath: str = None,  # The model to be analyzed
                 videoDatasetPath: str = None,  # The path with the whole dataset
                 stratifiedKFolds: bool = True,  # variable that identifies if stratified KFolds has been used
                 xTestPath: str = None,  # path containing the names of data used as test set
                 transformation: str = None,  # transformation applied to the original dataset
                 classesNames: list = None,  # name of classes, used for plotting the results
                 CNNModel: int = 2,  # CNN model to extract the features
                 params: dict = None,  # contains the parameters concerning the input size, the LSTM and Dense neurons
                 ):
        assert os.path.exists(videoDatasetPath), "Dataset does not exist"  # check if the dataset exist
        self.__dataPath = videoDatasetPath + '/{classname}/*.avi'  # used by keras video generator
        self.__globInputVideos = glob.glob(videoDatasetPath + "/*/*")  # get all the names of data
        self.__globInputVideos.sort()
        self.__labels = np.asarray(
            [int(i.split('/')[-2]) for i in self.__globInputVideos])  # get the labels of data automatically
        self.__classes = list(np.unique([i.split('/')[-2] for i in self.__globInputVideos]))  # get the classes
        self.__classes.sort()
        self.__nClasses = len(self.__classes)  # get the number of classes
        if params is not None:
            if "size" not in params.keys():
                print("Input Size missing: default assigned")
                self.__size = (15, 224, 224)  # default input size of the videos, the first dimension represents the frames,
                # the second and the third the dimension of the images. The number of channels will be automatically assigned later
            else:
                if not len(params["size"]) == 3:
                    print("Wrong input size: default assigned")
                    self.__size = (15, 224, 224)
                else:
                    self.__size = params["size"]
            if "LSTMNeurons" not in params.keys():
                print("LSTM Neurons set by default")
                self.__LSTMNeurons = 128  # number of neurons in the LSTM layer
            else:
                self.__LSTMNeurons = eval(params["LSTMNeurons"])
            if "DenseNeurons" not in params.keys():
                print("Dense Neurons set by default")
                self.__DenseNeurons = 64  # number of neurons in the LSTM layer
            else:
                self.__DenseNeurons = eval(params["DenseNeurons"])
        else:
            print("Input size default assigned")
            self.__size = (15, 224, 224)
            print("LSTM Neurons set by default")
            self.__LSTMNeurons = 128
            print("Dense Neurons set by default")
            self.__DenseNeurons = 64

        # check if the path of the test set exists
        assert os.path.exists(xTestPath), " Test path does not exist: provide the file with the data to be tested: " \
                                          "can be the folder path containing the data or a pickle file containing" \
                                          " the names of the data path to be tested"
        if xTestPath.endswith("pkl"):  # load the pickle file if exists
            with open(xTestPath, 'rb') as f:
                xTestDict = pkl.load(f)
                self.__X_test = xTestDict["X_Test"]
                self.__y_test = xTestDict["y_Test"]
                f.close()
        else:  # else get all the files that will be used as test set
            X_test = glob.glob(xTestPath + "/*/*")
            X_test.sort()
            self.__y_test = np.asarray([int(i.split('/')[-2]) for i in X_test])
            self.__X_test = []
            self.__X_test.append(X_test)

        # check if the models' path exists
        assert os.path.exists(modelPath), "Model path does not exist"
        self.__globModels = glob.glob(modelPath + '/*')

        self.__CNNModel = CNNModel
        if not 1 <= self.__CNNModel <= 9:
            print("CNN Model option not valid: default assigned")
            self.__CNNModel = 2

        # delete useless files from globModels
        indexToDel = []
        for i, m in enumerate(self.__globModels):
            if ".h5" not in m or "Model" + str(self.__CNNModel) not in m:
                indexToDel.append(i)
            else:
                if stratifiedKFolds:  # the names of the models must contain "Fold" string
                    if "Fold" not in m:
                        indexToDel.append(i)
                    if "OneFold" in m:
                        indexToDel.append(i)
                else:  # else remove the files that contain "Fold" string
                    if "OneFold" not in m:
                        indexToDel.append(i)
        for ele in sorted(indexToDel, reverse=True):
            del self.__globModels[ele]
        assert len(self.__globModels) > 0, "There are not models to test"
        self.__globModels.sort()

        listOfPossibleTrans = ["Canny", "Sobel_XY", "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]
        if transformation is None:  # if transformation is none assign the default
            print("Transformation assigned by default: No transformation")
            self.__transformation = "No_Trans"  # default option No Transformation
        elif transformation not in listOfPossibleTrans:  # if transformation is not in the list of possible transformation assign the default
            print("Requested transformation is not in the list of the possible transformation: No_Tran "
                  "assigned by default")
            self.__transformation = "No_Trans"  # default option No Transformation
        else:
            self.__transformation = transformation
        # check if transformation is in the model path, in the dataset path, and in the test data path
        assert self.__transformation in modelPath, "Input model path must contain the applied transformation"
        assert self.__transformation in self.__dataPath, "Dataset path must contain the applied transformation"
        assert self.__transformation in xTestPath, "Test data path must contain the applied transformation"

        # assign the classes names for plotting the results
        if classesNames is not None:
            self.__classesNames = classesNames
        else:
            print("Classes names assigned by default: Bed, Fall, Sit, Stand, Walk")
            self.__classesNames = ["Bed", "Fall", "Sit", "Stand", "Walk"]
        # and check if the list length is equal to the number of classes
        assert len(self.__classesNames) == len(
            self.__classes), "Length of classes names different from length of classes"

        if stratifiedKFolds:
            # if KFolds has been used, check if the models have a corresponding dataset to be tested
            if len(self.__X_test) == 1:
                warnings.warn(
                    " Number of datasets different from the number of folds, same dataset will be used to test each fold")
            else:
                assert len(self.__X_test) == len(
                    self.__globModels), " Number of datasets different from the number of folds"
            print("Test {} models on KFolds".format(self.__transformation))
        else:
            # else, check if there is only one list of data to be tested
            assert len(self.__X_test) == 1, " The list must have only one dimension"
            print("Test {} model without KFolds".format(self.__transformation))
        self.__stratifiedKFolds = stratifiedKFolds

        self.__channels = 1
        if transformation == "No_Trans":  # No Transformation is the only one with 3 channels (RGB)
            self.__channels = 3
        self.__size = self.__size + (self.__channels,)  # add the channel to the size of the input videos

    def testModels(self):  # function used to print the accuracy on the test set for each model contained in globModels
        testAccList = []
        for mod in self.__globModels:
            if not self.__stratifiedKFolds and "OneFold" not in mod:
                continue
            model = deep_network_test_and_tflite(self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
            model.load_weights(mod)
            print("Testing {}".format(mod.split('/')[-1]))
            if self.__stratifiedKFolds and len(self.__X_test) > 1:
                # take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.__X_test[indexTmp]
                yT = self.__y_test[indexTmp]
            else:
                xT = self.__X_test[0]
                yT = self.__y_test
            testGen = self.__generator(data_to_take=xT)  # create the generator for testing the model
            pred = model.predict(testGen, steps=len(xT))
            y_pred = np.argmax(pred, axis=1)
            true_labels = yT
            testAccTmp = 1 - np.count_nonzero(y_pred - true_labels) / len(y_pred)
            print('Test accuracy:', testAccTmp)
            testAccList.append(testAccTmp)
        testAccArray = np.asarray(testAccList)
        modelName = self.__globModels[0].split('_')[-1].split('.')[0]
        print("Average Accuracy {}: {} +- {}".format(modelName, np.mean(testAccArray), np.std(testAccArray)))
        return testAccList

    def confusion_matrix(self):  # Function used to plot the confusion matrix
        cm = np.zeros((self.__nClasses, self.__nClasses))
        if self.__stratifiedKFolds:
            print("Computing Confusion Matrix Stratified KFolds {}".format(self.__transformation))
        else:
            print("Computing Confusion Matrix {}".format(self.__transformation))
        for mod in self.__globModels:
            if not self.__stratifiedKFolds and "OneFold" not in mod:
                continue
            model = deep_network_test_and_tflite(self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
            model.load_weights(mod)
            if self.__stratifiedKFolds and len(self.__X_test) > 1:
                # take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.__X_test[indexTmp]
                yT = self.__y_test[indexTmp]
            else:
                xT = self.__X_test[0]
                yT = self.__y_test
            testGen = self.__generator(data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            y_pred = np.argmax(pred, axis=1)
            true_labels = yT
            cm += confusion_matrix(true_labels, y_pred)
            if not self.__stratifiedKFolds:  # If the KFolds technique is not used plot the confusion matrix for each model,
                # else sum the results for each fold
                self.__visualizeConfMatrix(cm)
                cm = np.zeros((self.__nClasses, self.__nClasses))  # Reinitialize the confusion matrix
        if self.__stratifiedKFolds:  # If the KFolds technique is used plot the cumulated confusion matrices
            self.__visualizeConfMatrix(cm)

    def computeROCandAUC(self, labelsOfPositive):  # function to plot the ROC and AUC curves. It takes in input the list
        # of the classes that must be considered as positive samples.
        # Initialization of dictionaries used to print the ROC and AUC curves
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Check if the list of positive is not empty
        assert len(labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.__nClasses, "the labels of positive samples are bigger than the number of classes"
        if self.__stratifiedKFolds:
            print("Computing ROC and AUC Curves Stratified KFolds {}".format(self.__transformation))
        else:
            print("Computing ROC and AUC Curves {}".format(self.__transformation))
        ii = 0
        for mod in self.__globModels:
            model = deep_network_test_and_tflite(self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
            model.load_weights(mod)
            if self.__stratifiedKFolds:
                # Take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.__X_test[indexTmp]
                yT = self.__y_test[indexTmp]
            else:
                xT = self.__X_test
                yT = self.__y_test
            testGen = self.__generator(data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            true_labels = yT
            fpr_tmp, tpr_tmp = self.__compute_ROC(pred, true_labels,
                                                  labelsOfPositive)  # Compute the false positive and true positive rates
            fpr[ii] = fpr_tmp
            tpr[ii] = tpr_tmp
            roc_auc[ii] = auc(fpr_tmp, tpr_tmp)  # Compute the AUC curve
            if not self.__stratifiedKFolds:  # If the KFolds technique is not used plot the ROC and AUC for each model, else plot the results for all the folds
                self.__visualizeROCandAUC(fpr, tpr, roc_auc)
            else:
                ii += 1
        if self.__stratifiedKFolds:
            self.__visualizeROCandAUC(fpr, tpr, roc_auc)

    def computeMetrics(self, labelsOfPositive):  # Function to compute al the metrics: precision, recall, specifity,
        # false positive rate, false negative rate, accuracy, and f1 score. It takes in input the list of the classes
        # that must be considered as positive samples.
        # Initialization of the results lists
        precision = []
        recall = []
        specifity = []
        FPR = []
        FNR = []
        accuracy = []
        f1score = []
        # Check if the list of positive is not empty
        assert len(labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.__nClasses, "the labels of positive samples are bigger than the number of classes"
        for i, mod in enumerate(self.__globModels):
            model = deep_network_test_and_tflite(self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
            model.load_weights(mod)
            if self.__stratifiedKFolds:
                # Take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.__X_test[indexTmp]
                yT = self.__y_test[indexTmp]
            else:
                xT = self.__X_test
                yT = self.__y_test
            testGen = self.__generator(data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            pred_labels = np.argmax(pred, axis=1)
            true_labels = yT
            predBin = []
            # Append 1 if the predictions ar contained in the positive labels list, else 0
            for jj in pred_labels:
                if jj in labelsOfPositive:
                    predBin.append(1)
                else:
                    predBin.append(0)
            predBin = np.asarray(predBin)
            true_labelsBin = []
            # Append 1 if the true labels ar contained in the positive labels list, else 0
            for jj in true_labels:
                if jj in labelsOfPositive:
                    true_labelsBin.append(1)
                else:
                    true_labelsBin.append(0)
            true_labelsBin = np.asarray(true_labelsBin)
            tn, fp, fn, tp = confusion_matrix(true_labelsBin,
                                              predBin).ravel()  # compute the true positive, the false positive,
            # the true negative, and false negative rates.
            # Append the results to the corresponding lists
            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))
            specifity.append(tn / (tn + fp))
            FPR.append(fp / (fp + tn))
            FNR.append(fn / (fn + tp))
            accuracy.append((tp + tn) / (tp + tn + fn + fp))
            f1score.append(2 * tp / (2 * tp + fp + fn))
        # Create a dictionary for the results
        Metrics = {"Prec": np.mean(precision), "Rec": np.mean(recall), "Spec": np.mean(specifity), "FPR": np.mean(FPR),
                   "FNR": np.mean(FNR), "Acc": np.mean(accuracy), "F1": np.mean(f1score)}
        fileName = "../Results/" + self.__transformation + "/" + self.__transformation + "_Metrics_Model{}.txt".format(
                self.__CNNModel)
        # Save the results in a file
        print("Metrics have been save at ", fileName)
        with open(fileName, 'w') as f:
            for key, value in Metrics.items():
                f.write('%s:%.3f\n' % (key, float(value)))
        self.__visualizeMetrics(Metrics)

    def __visualizeMetrics(self, metrics):
        # Function for visualizing the metrics as a bar-plot"
        prec = metrics["Prec"]
        rec = metrics["Rec"]
        spec = metrics["Spec"]
        fpr = metrics["FPR"]
        fnr = metrics["FNR"]
        acc = metrics["Acc"]
        f1 = metrics["F1"]
        met = ["Acc", "Prec", "Rec", "Spec", "FPR", "FNR", "F1"]
        metValues = [acc, prec, rec, spec, fpr, fnr, f1]
        data = {"Names": met, "Values": metValues}
        df = pd.DataFrame(data, columns=['Names', 'Values'])
        plt.figure(figsize=(12, 8))
        plots = sns.barplot(x="Names", y="Values", data=df)
        for bar in plots.patches:
            plots.annotate(format(bar.get_height(), '.3f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='center',
                           size=15, xytext=(0, 8),
                           textcoords='offset points')
        plt.xlabel("Metrics", size=28)
        plt.ylabel("Values", size=28)
        plt.ylim((0, 1.05))
        plt.yticks(np.arange(0, 1.05, 0.2))
        plt.axhline(y=1, color='r', linestyle='--', linewidth=6)
        title = "Metrics Model {} {}".format(self.__CNNModel, self.__transformation)
        plt.title(title, fontsize=28)
        plt.tick_params(axis='x', labelsize=24)
        plt.tick_params(axis='y', labelsize=24)
        plt.show()

    def __visualizeConfMatrix(self, cm):  # Function plotting the confusion matrix
        cm_df = pd.DataFrame(cm, index=self.__classesNames, columns=self.__classesNames)
        plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

        sns.set(font_scale=2.5)
        s = np.sum(cm, axis=1)
        sns.heatmap(cm_df / s, annot=cm, cmap="Greys", annot_kws={"size": 36}, cbar=False, linewidths=1,
                    linecolor='black',
                    fmt='g', square=True)
        plt.ylabel('True labels', fontsize=28, fontweight="bold")
        plt.xlabel('Predicted label', fontsize=28, fontweight="bold")

        if "Model" in self.__globModels[0]:
            nameModel = " Model" + self.__globModels[0][self.__globModels[0].find("Model") + 5]
        else:
            nameModel = ""
        nameTran = self.__transformation
        if self.__transformation == "No_Trans_Gray":
            nameTran = "Gray"
        if self.__stratifiedKFolds:
            title = "Conf. Matrix KFolds " + nameTran + nameModel
        else:
            title = "Conf Matrix " + nameTran + nameModel
        plt.title(title, fontsize=32, fontweight="bold")
        plt.show()

    def __compute_ROC(self, y_score, true_labels, labelsOfPositive):
        cl = np.asarray([int(i) for i in self.__classes])
        y_test = label_binarize(true_labels, classes=cl)
        y_test_tmp = []
        y_score_tmp = []
        for i in labelsOfPositive:
            y_test_tmp = np.concatenate((y_test_tmp, y_test[:, i]))
            y_score_tmp = np.concatenate((y_score_tmp, y_score[:, i]))
        fpr, tpr, _ = roc_curve(y_test_tmp, y_score_tmp, drop_intermediate=False)
        return fpr, tpr

    def __visualizeROCandAUC(self, fpr, tpr, roc_auc):  # Function plotting the ROC and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(fpr))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(fpr)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(fpr)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"], label="Average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]),
                 color="black", linestyle=":", linewidth=5)

        colors = ["aqua", "darkorange", "cornflowerblue", "deeppink", "limegreen", "darksalmon", "orchid", "gold",
                  "grey",
                  "navy"]
        for i in range(len(fpr) - 1):
            c = i % len(colors)
            if self.__stratifiedKFolds:
                plt.plot(fpr[i], tpr[i], color=colors[c], linewidth=5,
                         label="ROC curve of Fold {0} (area = {1:0.3f})".format(i, roc_auc[i]))
            else:
                plt.plot(fpr[i], tpr[i], color=colors[c], linewidth=5,
                         label="ROC curve (area = {0:0.3f})".format(roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", lw=5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=36)
        plt.ylabel("True Positive Rate", fontsize=36)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        if "Model" in self.__globModels[0]:
            nameModel = " Model" + self.__globModels[0][self.__globModels[0].find("Model") + 5]
        else:
            nameModel = ""
        nameTran = self.__transformation
        if self.__transformation == "No_Trans_Gray":
            nameTran = "Gray"
        if self.__stratifiedKFolds:
            title = "ROC and AUC KFolds " + nameTran + nameModel
        else:
            title = "ROC and AUC " + nameTran + nameModel
        plt.title(title, fontsize=40)
        plt.legend(loc="lower right", fontsize=24)
        plt.show()

    def __generator(self, data_to_take):  # function that returns the keras video generator
        gen = VideoFrameGenerator(rescale=1 / 255.,
                                  classes=self.__classes,
                                  glob_pattern=self.__dataPath,
                                  nb_frames=self.__size[0],
                                  shuffle=False,
                                  batch_size=1,
                                  target_shape=self.__size[1:3],
                                  nb_channel=self.__channels,
                                  use_frame_cache=True,
                                  _test_data=data_to_take)
        return gen
