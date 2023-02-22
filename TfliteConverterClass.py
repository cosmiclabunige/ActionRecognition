import shutil
import cv2.cv2 as cv2
import glob
import os
import numpy as np
from CNN_architectures import *



class TfliteConverterClass:
    """
    Class to convert the h5 models in tflite.
    The input parameters are:

    - *modelPath2Convert*: the path containing the h5 models to be converted
    - *datasetPath*: the path of the dataset used for the representative dataset
    - *transformation*: the transformation applied to the input dataset, it must be compliant with the model path name and the dataset path name
    - *CNNModel*: model that has to be trained (default is model 2). Possible choices are in the range 1-3

    """
    def __init__(self,
                 modelPath2Convert: str = None,  # the input path containing the h5 models to be converted
                 datasetPath: str = None,  # the dataset used to create the representative dataset
                 transformation: str = None,  # the transformation applied to the original data
                 CNNModel: int = 2,  # CNN Model to be converted
                 params: dict = None,  # contains the parameters concerning the input size, the LSTM and Dense neurons
                 ):
        if params is not None:
            if "nClasses" not in params.keys():
                self.__nClasses = 5
            else:
                self.__nClasses = eval(params["nClasses"])
            if "size" not in params.keys():
                print("Input Size missing: default assigned")
                self.__size = (
                15, 224, 224)  # default input size of the videos, the first dimension represents the frames,
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
            print("nClasses default assigned")
            self.__nClasses = 5
            print("Input size default assigned")
            self.__size = (15, 224, 224)
            print("LSTM Neurons set by default")
            self.__LSTMNeurons = 128
            print("Dense Neurons set by default")
            self.__DenseNeurons = 64

        # check if the model path to be converted is not none and exists
        assert modelPath2Convert is not None, "Model path to be converted is none"
        assert os.path.exists(modelPath2Convert), "Model Path does not exist"
        self.__globModels = glob.glob(modelPath2Convert + '/*')  # take all the files in the path

        self.__CNNModel = CNNModel
        if not 1 <= self.__CNNModel <= 9:
            print("CNN Model option not valid: default assigned")
            self.__CNNModel = 2

        indexToDel = []
        # Remove all the models that are not "h5"
        for i, m in enumerate(self.__globModels):
            if "h5" not in m or "Model"+str(self.__CNNModel) not in m:
                indexToDel.append(i)
        for ele in sorted(indexToDel, reverse=True):
            del self.__globModels[ele]
        assert len(self.__globModels) > 0, "Not models to convert"  # check if the list of model is not empty
        self.__globModels.sort()

        self.__convertedModelPath = "../ModelsTFLite/" + transformation  # assign by default where the converted models will be saved
        if not os.path.exists(self.__convertedModelPath):
            os.makedirs(self.__convertedModelPath)
        # check if the dataset path exists
        assert os.path.exists(datasetPath), "Dataset Path does not exist"
        self.__dataPath = datasetPath

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
        # check if transformation is in the model path and in the dataset path
        assert self.__transformation in modelPath2Convert, "Input model path must contain the applied transformation"
        assert self.__transformation in self.__dataPath, "Dataset path must contain the applied transformation"

        self.__modelDirTmp = '../ModelsTmp'  # assign by default the folder for the temporary models
        if not os.path.exists(self.__modelDirTmp):
            os.mkdir(self.__modelDirTmp)

    def create_tflite_fp32(self):  # convert in the tflite fp32 bit model
        for m in self.__globModels:
            self.__save_temporary_model(m)  # save the h5 model as pb
            name = (m.split('/')[-1]).split('.')[0]  # extract automatically the name of the model
            modelDirTmp = os.path.join(self.__modelDirTmp, name)
            converter = tf.lite.TFLiteConverter.from_saved_model(modelDirTmp)  # initialize the converter object
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            nameFull = os.path.join(self.__convertedModelPath, name + "_fp32.tflite")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float32]  # specify the representation of the operations inside the tflite model
            # invoke the conversion and save the tflite model
            tflite_modelFP32 = converter.convert()
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelFP32)
            print("TFLITE FP32 MODEL {} CREATED".format(name))

    def create_tflite_fp16(self):  # convert in the tflite fp16 bit model
        for m in self.__globModels:
            self.__save_temporary_model(m)  # save the h5 model as pb
            name = (m.split('/')[-1]).split('.')[0]  # extract automatically the name of the model
            modelDirTmp = os.path.join(self.__modelDirTmp, name)
            converter = tf.lite.TFLiteConverter.from_saved_model(modelDirTmp)  # initialize the converter object
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            nameFull = os.path.join(self.__convertedModelPath, name + "_fp16.tflite")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.target_spec.supported_types = [tf.float16]  # specify the representation of the operations inside the tflite model
            tflite_modelFP16 = converter.convert()
            # invoke the conversion and save the tflite model
            with open(nameFull, 'wb') as f:
                f.write(tflite_modelFP16)
            print("TFLITE FP16 MODEL {} CREATED".format(name))

    def __save_temporary_model(self, m):  # function used to create a pb file of the model to be converted
        neomodel = deep_network_test_and_tflite(self.__CNNModel, self.__size, self.__LSTMNeurons, self.__DenseNeurons, self.__nClasses)
        neomodel.load_weights(m)
        neomodel.summary()

        # create the concrete function for the conversion of the h5 model in pb format
        run_model = tf.function(lambda x: neomodel(x))
        shape = np.asarray([neomodel.inputs[0].shape[i] for i in range(1, len(neomodel.inputs[0].shape))])
        shape = np.concatenate(([1], shape))
        concrete_func = run_model.get_concrete_function(tf.TensorSpec(shape, neomodel.inputs[0].dtype))
        modelDirTmp = os.path.join(self.__modelDirTmp, (m.split('/')[-1]).split('.')[0])
        if os.path.exists(modelDirTmp):
            shutil.rmtree(modelDirTmp)
            os.mkdir(modelDirTmp)
        else:
            os.mkdir(modelDirTmp)
        # convert the model in pb format
        neomodel.save(modelDirTmp, save_format="tf", signatures=concrete_func)
        print("Tmp folder for Model {} conversion created!".format(m.split('/')[-1]))
