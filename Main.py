from ImagesTransformationClass import ImagesTransformationClass
from TrainVideoClassifierClass import TrainVideoClassifierClass
from TfliteConverterClass import TfliteConverterClass
from AnalyzeResultsClass import AnalyzeResultsClass

inputImagesPath = "../Dataset/Images" # Dataset path for training
transformation = "No_Trans_Gray" # transformation
model = 2 # model architecture to train
videosOutputPath = "../Dataset/Videos_" + transformation # path where save the sequences
nFolds = 5 # number of folds for the training
inputVideoPath = videosOutputPath # the path of the sequences with which the model will be trained
modelPath = "../Results/" + transformation # model path for the analysis of the results

# Covert the images in sequences applying transformation
ImagesTransformationClass(imagesInputPath=inputImagesPath, transformation=transformation, videosOutputPath=videosOutputPath)
# Train the model
TVC = TrainVideoClassifierClass(videoDatasetPath=inputVideoPath, transformation=transformation, nFolds=nFolds, CNNModel=model)
TVC.training()

# Change the tested dataset based on the number of folds
if nFolds > 1:
    stratifiedKFolds = True
    xTestpath = "../Results/" + transformation + "/X_Test_" + transformation + "_StratifiedKFolds_Model{}.pkl".format(model)
else:
    stratifiedKFolds = False
    xTestpath = "../Results/" + transformation + "/X_Test_" + transformation + "_OneFold_Model{}.pkl".format(model)

# Test the model, compute the confusion matrix, plot the ROC and AUC curves, compute the metrics
ARC = AnalyzeResultsClass(modelPath=modelPath, videoDatasetPath=inputVideoPath, stratifiedKFolds=stratifiedKFolds, transformation=transformation, xTestPath=xTestpath, CNNModel=model)
ARC.testModels()
ARC.confusion_matrix()
ARC.computeROCandAUC([0, 1])
ARC.computeMetrics([0, 1])

# Convert the model in FP16 and FP32 TFLITE format
TFLiteConv = TfliteConverterClass(transformation=transformation, modelPath2Convert=modelPath, datasetPath=inputVideoPath, CNNModel=model)
TFLiteConv.create_tflite_fp32()
TFLiteConv.create_tflite_fp16()
