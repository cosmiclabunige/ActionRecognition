# Radar Video Action Recognition

![Logo](https://www.drupal.org/files/5_21.png)



## Authors

- [**Christian Gianoglio**](https://www.linkedin.com/in/christian-gianoglio/)
- [**Ammar Mohanna**](https://www.linkedin.com/in/ammar-mohanna/)
- [Ali Rizik](https://www.linkedin.com/in/ali-rizik/)
- [Laurence Moroney](https://www.linkedin.com/in/laurence-moroney/)
- [Maurizio Valle](https://www.linkedin.com/in/maurizio-valle-22ba5014/)

## Introduction

This repository contains the code used during our research process, leading to the publication titled:

> "On Edge Human Action Recognition Using Radar-Based Sensing and Deep Learning."

In this paper, we propose a radar-based human action recognition system capable of real-time action recognition. Range-Doppler maps extracted from a low-cost FMCW radar are processed by a deep neural network, and the system is designed for edge deployment. The results demonstrate the system's ability to recognize five human actions with an accuracy of 93.2% and an inference time of 2.95s. Additionally, the system's performance in binary classification (fall vs. non-fall actions) is assessed, achieving an accuracy of 96.8% with a low false-negative rate of 4%. To optimize the trade-off between accuracy and computational cost, we measure the energy precision ratio of the system deployed on the edge, achieving a value of 1.04, with an ideal ratio close to zero.

![Proposed Neural Network](https://user-images.githubusercontent.com/32446816/181509833-d30ea2ea-fbd6-4a38-b20f-cc7be8428f52.png)

## Dataset

The datasets used in our research can be found here:

- [Original Dataset](https://www.kaggle.com/datasets/cosmiclab/actionrecognition)
- [Additional Test Dataset in a New Environment](https://www.kaggle.com/datasets/cosmiclab/actionrecognitionnewenvinronment)

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8
- Tensorflow 2.8
- keras-video-generators 1.0.14
- tflite-runtime 2.7 (Optional)


### Installation and Usage

To set up and run the code for action recognition, follow these steps:

1. Create a Conda environment (if not already installed):

   ```  
   conda create -n action_recognition
   ```

2. Activate the Conda environment:
    
   ```
   conda activate action_recognition
   ```

3. Upgrade pip to ensure you have the latest version:

   ```
   python -m pip install --upgrade pip
   ```

4. Install the required Python packages using pip:

   ```
   pip install tensorflow-cpu==2.8.0
   pip install keras-video-generators==1.0.14
   ```

5. Once you have the dependencies installed, you can run the main script:

   ```
   python main.py
   ```

### Usage

1.1. Download the data from the Kaggle repositories mentioned above. The second repository has been used only for testing purposes.

   ![Kaggle Repositories](https://user-images.githubusercontent.com/32446816/181509190-3cc9ee4f-1f6c-4946-b5a3-14490467251c.png)

1.2. Place the dataset in the proper place:

# INSERT DATASET FOLDER STRUCTURE SCREENSHOT

2. The `main.py` script imports all other scripts described below. It includes examples to transform images, create sequences of transformed images with a .avi extension, train one of the available models, analyze the results, and convert the model to TFLITE format for deployment.

3. The `ImageTransformation` script generates the five different image transformations mentioned in Section 2.B of the paper.

   ![Image Transformations](https://user-images.githubusercontent.com/32446816/181508622-bb9d617a-c0fb-455a-8b84-7bb6b8bd0685.png)

4. The `TrainVideoClassifier` script loads data and trains the chosen model.

5. After training, the `AnalyzeResults` script is responsible for generating assessment metrics computed in the paper (e.g., confusion matrix, ROC and AUC, metrics, etc.).

   ![Confusion Matrix and ROC](https://user-images.githubusercontent.com/32446816/181509578-7c085fcc-e9c4-46ae-9eca-55aaf94498cb.png)

6. To deploy on an edge device, you can use the `ConvertTFLITE` to convert your trained model to TFLITE-compatible format.

## Questions & Inquiries

For any questions or inquiries, feel free to contact us directly:

- [Christian Gianoglio](mailto:christian.gianoglio@unige.it)
- [Ammar Mohanna](https://www.linkedin.com/in/ammar-mohanna/)
