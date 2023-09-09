try:
    from action_recognition.train import TrainVideoClassifier
    from action_recognition.convert_tflite import ConvertTFLITE
    from action_recognition.analyze_results import AnalyzeResults
    from action_recognition.image_transformation import ImagesTransformation
except Exception as e:
    print('Error loading modules in main.py: ', e)


def main():
    # Define input and output paths
    inputImagesPath = "./Dataset/Images"  # Dataset path for training
    transformation = "No_Trans_Gray"  # Transformation
    model = 2  # Model architecture to train

    # Path to save the sequences
    videosOutputPath = f"../Dataset/Videos_{transformation}"
    nFolds = 5  # Number of folds for training
    inputVideoPath = videosOutputPath  # Path of the sequences for training
    
    # Model path for result analysis
    modelPath = f"../Results/{transformation}"

    # Convert the images into sequences applying the transformation
    ImagesTransformation(
        imagesInputPath=inputImagesPath,
        transformation=transformation,
        videosOutputPath=videosOutputPath
    )

    # Train the model
    TVC = TrainVideoClassifier(
        videoDatasetPath=inputVideoPath,
        transformation=transformation,
        nFolds=nFolds,
        CNNModel=model
    )
    TVC.training()

    # Determine the path for testing based on the number of folds
    if nFolds > 1:
        stratifiedKFolds = True
        xTestpath = f"../Results/{transformation}/X_Test_{transformation}_StratifiedKFolds_Model{model}.pkl"
    else:
        stratifiedKFolds = False
        xTestpath = f"../Results/{transformation}/X_Test_{transformation}_OneFold_Model{model}.pkl"

    # Test the model, compute the confusion matrix, plot the ROC and AUC curves, compute the metrics
    ARC = AnalyzeResults(
        modelPath=modelPath,
        videoDatasetPath=inputVideoPath,
        stratifiedKFolds=stratifiedKFolds,
        transformation=transformation,
        xTestPath=xTestpath,
        CNNModel=model
    )
    ARC.test_models()
    ARC.confusion_matrix()
    ARC.compute_ROC_AUC([0, 1])
    ARC.compute_metrics([0, 1])

    # Convert the model to FP16 and FP32 TFLITE formats
    TFLiteConv = ConvertTFLITE(
        transformation=transformation,
        modelPath2Convert=modelPath,
        datasetPath=inputVideoPath,
        CNNModel=model
    )
    TFLiteConv.create_tflite_fp32()
    TFLiteConv.create_tflite_fp16()


if __name__ == "__main__":
    main()
