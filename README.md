# Edge Action Recognition
This work presented in the folders was used during a research process.
It resulted the publication titled: 

> On Edge Human Action Recognition Using Radar-Based Sensing and Deep Learning

 In this paper, we propose a radar-based human action recognition system, capable of recognizing actions in real-time. Range-Doppler maps, extracted from a low-cost FMCW radar, are fed into a deep neural network. The system is deployed on an edge device. The results show that the system can recognize five human actions with an accuracy of 93.2% and an inference time of 2.95s. Raising an alarm when a harmful action happens is a crucial feature in an indoor safety application. Thus, the performance during the binary classification, i.e. fall vs non-fall actions, is also assessed, achieving an accuracy of 96.8% with a false-negative rate of 4%. To find the best trade-off between accuracy and computational cost, the energy precision ratio of the system deployed on the edge is measured. The system achieves a 1.04 energy precision ratio value, where an ideal ratio would be close to zero.
 
The proposed deep neural network for five human actions recognition is:

![NN](https://user-images.githubusercontent.com/32446816/181509833-d30ea2ea-fbd6-4a38-b20f-cc7be8428f52.png)

You can find the data here that were collected in the reseach process here: [https://www.kaggle.com/datasets/cosmiclab/actionrecognition](https://www.kaggle.com/datasets/cosmiclab/actionrecognition)

Moreover, we collected another test set in a new envinronment. You can find the data here: [https://www.kaggle.com/datasets/cosmiclab/actionrecognitionnewenvinronment](https://www.kaggle.com/datasets/cosmiclab/actionrecognitionnewenvinronment)

### Here is a quick overview of the files included in this github repository

All the code has been written in python. The following packages and interpreter have been used:
- Python 3.8
- Tensorflow 2.8
- keras-video-generators 1.0.14
- tflite-runtime 2.7

1. Download the data from the aforementioned kaggle repositories. In the experiments the second repository has been used only for testing
![kaggle_repo](https://user-images.githubusercontent.com/32446816/181509190-3cc9ee4f-1f6c-4946-b5a3-14490467251c.png)
<br/><br/>

2. The script *Main.py* import all the other scripts described in the following. It includes an example to transform the images and create sequences of transformed images with .avi extension, train one of the available models, analyze the results, convert the model in tflite for deployment.

2. The script *ImageTransformationClass* is responsible for generating the 5 different image transformations mentioned in Section 2.B of the paper
<br/><br/>
![Untitled](https://user-images.githubusercontent.com/32446816/181508622-bb9d617a-c0fb-455a-8b84-7bb6b8bd0685.png)
<br/><br/>

3. The script *TrainVideoClassifierClass* is in charge of loading data and training the chosen model
<br/><br/>

4. After training, the script *AnalyzeResultsClass* is responsible for generating the assessments metrics computed in the paper (e.g., confusion matrix, ROC and AUC, metrics, etc.)
![confroc](https://user-images.githubusercontent.com/32446816/181509578-7c085fcc-e9c4-46ae-9eca-55aaf94498cb.png)
<br/><br/>

5. Finally, to deploy on an edge device, you can use the *TfliteConverterClass* to change the format of your trained model to TFLITE compatible.

<br/><br/>
<br/><br/>

For any questions, feel free to ask your questions or contact us directly:

*christian.gianoglio@unige.it*
