try:
    import os
    import glob
    import warnings
    import numpy as np
    import pandas as pd
    import pickle as pkl
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    from action_recognition.cnn_arch import *
    from keras_video import VideoFrameGenerator
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, confusion_matrix

    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
except Exception as e:
    print('Error loading modules in analyze_results.py: ', e)


class AnalyzeResults:
    """
    This class is used to show the results concerning the accuracy of the model on the test data, 
    and to plot the confusion matrices and ROC and AUC curves.
    """

    def __init__(self, modelPath=None, videoDatasetPath=None, stratifiedKFolds=True, xTestPath=None, transformation=None, classesNames=None, CNNModel=2, params=None):
        """
        Initialize the AnalyzeResults class with parameters.

        Args:
            modelPath (str): The path where model to be tested is saved.
            videoDatasetPath (str): The dataset path used for plotting the results.
            stratifiedKFolds (bool): If True, models trained with KFolds strategy will be analyzed.
            xTestPath (str): The path/file containing the names of the data that form the test set.
            transformation (str): The transformation applied to the original dataset.
            classesNames (list): List containing the names of the classes for plotting the results.
            CNNModel (int): Model on which to compute the test accuracy, metrics, confusion matrices (default is model 2).
            params (dict): Parameters concerning the input size, the LSTM, and Dense neurons.
        """
        assert os.path.exists(videoDatasetPath), "Dataset does not exist"  # check if the dataset exist
        self.__dataPath = videoDatasetPath / '{classname}' / '*.avi'  # used by keras video generator
        videos = videoDatasetPath.rglob("**/*.avi")
        self.__globInputVideos = [match for match in videos if match.is_file()]
        # automatic labels extraction
        labels = []
        for files in self.__globInputVideos:
            lab = files.parent.name
            try: 
                labels.append(int(lab))
            except:
                continue
        self.__labels = np.asarray(labels)
        self.__classes = list(np.unique([i.name for i in videoDatasetPath.glob('*')]))  # classes of the dataset
        self.__classes.sort()
        self.__nClasses = len(self.__classes)  # number of classes

        if params is not None:
            if "size" not in params.keys():
                print("Input Size missing: default assigned")
                # default input size of the videos, the first dimension represents the frames,
                self.__size = (15, 224, 224)
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
        if str(xTestPath).endswith("pkl"):  # load the pickle file if exists
            with open(xTestPath, 'rb') as f:
                xTestDict = pkl.load(f)
                self.__X_test = xTestDict["X_Test"]
                self.__y_test = xTestDict["y_Test"]
                f.close()
        else:  # else get all the files that will be used as test set
            X_test = Path(xTestPath).glob("**/*.avi")
            X_test = [str(match) for match in X_test if match.is_file()]
            labels = []
            for files in self.__globInputVideos:
                lab = files.parent.name
                try: 
                    labels.append(int(lab))
                except:
                    continue
            self.__y_test = np.asarray(labels)
            self.__X_test = []
            self.__X_test.append(X_test)

        # check if the models' path exists
        assert os.path.exists(modelPath), "Model path does not exist"
        models = modelPath.glob("*")
        self.__globModels = [str(match) for match in models if match.is_file()]

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

        listOfPossibleTrans = ["Canny", "Sobel_XY",
                               "Roberts", "Binary", "No_Trans_Gray", "No_Trans"]
        if transformation is None:  # if transformation is none assign the default
            print("Transformation assigned by default: No transformation")
            self.__transformation = "No_Trans"  # default option No Transformation
        # if transformation is not in the list of possible transformation assign the default
        elif transformation not in listOfPossibleTrans:
            print("Requested transformation is not in the list of the possible transformation: No_Tran "
                  "assigned by default")
            self.__transformation = "No_Trans"  # default option No Transformation
        else:
            self.__transformation = transformation
        # check if transformation is in the model path, in the dataset path, and in the test data path
        assert self.__transformation in str(modelPath), "Input model path must contain the applied transformation"
        assert self.__transformation in str(self.__dataPath), "Dataset path must contain the applied transformation"
        assert self.__transformation in str(xTestPath), "Test data path must contain the applied transformation"

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
            assert len(
                self.__X_test) == 1, " The list must have only one dimension"
            print("Test {} model without KFolds".format(self.__transformation))
        self.__stratifiedKFolds = stratifiedKFolds

        self.__channels = 1
        # No Transformation is the only one with 3 channels (RGB)
        if transformation == "No_Trans":
            self.__channels = 3
        # add the channel to the size of the input videos
        self.__size = self.__size + (self.__channels,)

    def test_models(self):
        """
        Function used to print the accuracy on the test set for each model contained in globModels.
        """
        testAccList = []
        for mod in self.__globModels:
            if not self.__stratifiedKFolds and "OneFold" not in mod:
                continue
            model = deep_network_test_and_tflite(
                self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
            model.load_weights(mod)
            print("Testing {}".format(mod))
            if self.__stratifiedKFolds and len(self.__X_test) > 1:
                # take the correct dataset to test the model along the folds
                indexTmp = int(mod[mod.find("Fold") + 4])
                xT = self.__X_test[indexTmp]
                yT = self.__y_test[indexTmp]
            else:
                xT = self.__X_test[0]
                yT = self.__y_test
            # create the generator for testing the model
            testGen = self.__generator(data_to_take=xT)
            pred = model.predict(testGen, steps=len(xT))
            y_pred = np.argmax(pred, axis=1)
            true_labels = yT
            testAccTmp = 1 - \
                np.count_nonzero(y_pred - true_labels) / len(y_pred)
            print('Test accuracy:', testAccTmp)
            testAccList.append(testAccTmp)
        testAccArray = np.asarray(testAccList)
        modelName = self.__globModels[0].split('_')[-1].split('.')[0]
        print("Average Accuracy {}: {} +- {}".format(modelName,
              np.mean(testAccArray), np.std(testAccArray)))
        return testAccList

    def confusion_matrix(self):
        """
        Function used to plot the confusion matrix.
        """
        cm = np.zeros((self.__nClasses, self.__nClasses))
        if self.__stratifiedKFolds:
            print("Computing Confusion Matrix Stratified KFolds {}".format(
                self.__transformation))
        else:
            print("Computing Confusion Matrix {}".format(self.__transformation))
        for mod in self.__globModels:
            if not self.__stratifiedKFolds and "OneFold" not in mod:
                continue
            model = deep_network_test_and_tflite(
                self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
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
                # Reinitialize the confusion matrix
                cm = np.zeros((self.__nClasses, self.__nClasses))
        if self.__stratifiedKFolds:  # If the KFolds technique is used plot the cumulated confusion matrices
            self.__visualizeConfMatrix(cm)

    def compute_ROC_AUC(self, labelsOfPositive):
        """
        Function to compute ROC and AUC curves.

        Args:
            labelsOfPositive (list): List of classes to consider as positive samples.
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # Check if the list of positive is not empty
        assert len(
            labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.__nClasses, "the labels of positive samples are bigger than the number of classes"
        if self.__stratifiedKFolds:
            print("Computing ROC and AUC Curves Stratified KFolds {}".format(
                self.__transformation))
        else:
            print("Computing ROC and AUC Curves {}".format(self.__transformation))
        ii = 0
        for mod in self.__globModels:
            model = deep_network_test_and_tflite(
                self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
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

    def compute_metrics(self, labelsOfPositive):
        """
        Function to compute metrics: precision, recall, specificity, FPR, FNR, accuracy, and f1 score.

        Args:
            labelsOfPositive (list): List of classes to consider as positive samples.
        """
        precision = []
        recall = []
        specifity = []
        FPR = []
        FNR = []
        accuracy = []
        f1score = []
        # Check if the list of positive is not empty
        assert len(
            labelsOfPositive) > 0, "list of positive classes must be not empty"
        for i in labelsOfPositive:
            # Check if the positive labels are correct
            assert i < self.__nClasses, "the labels of positive samples are bigger than the number of classes"
        for i, mod in enumerate(self.__globModels):
            model = deep_network_test_and_tflite(
                self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
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
        fileName = Path("results") / self.__transformation / f"{self.__transformation}_Metrics_Model{self.__CNNModel}.txt"    
        # Save the results in a file
        print("Metrics have been save at ", fileName)
        with open(fileName, 'w') as f:
            for key, value in Metrics.items():
                f.write('%s:%.3f\n' % (key, float(value)))
        self.__visualizeMetrics(Metrics)

    def __visualizeMetrics(self, metrics):
        """
        Function for visualizing the metrics as a bar-plot.

        Args:
            metrics.
        """
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
        title = "Metrics Model {} {}".format(
            self.__CNNModel, self.__transformation)
        plt.title(title, fontsize=28)
        plt.tick_params(axis='x', labelsize=24)
        plt.tick_params(axis='y', labelsize=24)
        plt.show()

    def __visualizeConfMatrix(self, cm):
        """
        Function plotting the confusion matrix.

        Args:
            cm.
        """
        cm_df = pd.DataFrame(cm, index=self.__classesNames,
                             columns=self.__classesNames)
        plt.figure(num=None, figsize=(10, 10), dpi=80,
                   facecolor='w', edgecolor='k')

        sns.set(font_scale=2.5)
        s = np.sum(cm, axis=1)
        sns.heatmap(cm_df / s, annot=cm, cmap="Greys", annot_kws={"size": 36}, cbar=False, linewidths=1,
                    linecolor='black',
                    fmt='g', square=True)
        plt.ylabel('True labels', fontsize=28, fontweight="bold")
        plt.xlabel('Predicted label', fontsize=28, fontweight="bold")

        if "Model" in self.__globModels[0]:
            nameModel = " Model" + \
                self.__globModels[0][self.__globModels[0].find("Model") + 5]
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
        """
        Function compute ROC.

        Args:
            y_score, true_labels, labelsOfPositive.
        """
        cl = np.asarray([int(i) for i in self.__classes])
        y_test = label_binarize(true_labels, classes=cl)
        y_test_tmp = []
        y_score_tmp = []
        for i in labelsOfPositive:
            y_test_tmp = np.concatenate((y_test_tmp, y_test[:, i]))
            y_score_tmp = np.concatenate((y_score_tmp, y_score[:, i]))
        fpr, tpr, _ = roc_curve(y_test_tmp, y_score_tmp,
                                drop_intermediate=False)
        return fpr, tpr

    def __visualizeROCandAUC(self, fpr, tpr, roc_auc):
        """
        Function plotting the ROC and AUC.

        Args:
            cm.
        """
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
            nameModel = " Model" + \
                self.__globModels[0][self.__globModels[0].find("Model") + 5]
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

    def __generator(self, data_to_take):
        """
        Function that returns the keras video generator.

        Args:
            data_to_take: Data to use for generating sequences.
        """
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
