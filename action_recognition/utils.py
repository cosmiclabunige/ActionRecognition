try:
    import os
    import cv2
    import pickle
    import shutil
    import numpy as np
    from pathlib import Path
    import tensorflow as tf
    from scipy import ndimage
except Exception as e:
    print('Error loading modules in utils.py: ', e)


class LearningRateReducerCb(tf.keras.callbacks.Callback):
    # class for reducing the learning rate every 5 epochs, it's called automatically by the fit function
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr * 0.9
            self.model.optimizer.lr.assign(new_lr)


class LearningRateReducerCbFineTuning(tf.keras.callbacks.Callback):
    # class for reducing the learning rate in the fine-tuning procedure every 7 epochs, it's called automatically by the fit function
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 7 == 0:
            old_lr = self.model.optimizer.lr.read_value()
            new_lr = old_lr * 0.5
            self.model.optimizer.lr.assign(new_lr)

# Canny Transformation


def CannyTrans(imPath):
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image=img_gray, threshold1=30, threshold2=30)
    return canny


# Sobel XY Transformation
def SobelXY(imPath):
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    sobelXY = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1,
                        ksize=5)
    return sobelXY


# Robert Transformation
def Roberts(imPath):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])
    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    mg = cv2.imread(imPath)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    img_gr = img_gray / 255.0
    vertical = ndimage.convolve(img_gr, roberts_cross_v)
    horizontal = ndimage.convolve(img_gr, roberts_cross_h)
    roberts = np.sqrt(np.square(horizontal) + np.square(vertical))
    roberts *= 255
    return roberts


# Binary Transformation
def Binary(imPath):
    mg = cv2.imread(imPath)
    img_gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    pixel_values = img_gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, 5, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_gray.shape)
    binary = ndimage.median_filter(segmented_image, size=30)
    minn = np.min(binary)
    binary[binary < minn + 1] = 0
    binary[binary >= minn + 1] = 255
    return binary


# No Transformation, images saved in gray scale
def NoTransGray(imPath):
    mg = cv2.imread(imPath)
    gray = cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)
    return gray


# No Transformation
def NoTrans(imPath):
    noTrans = cv2.imread(imPath)
    return noTrans
