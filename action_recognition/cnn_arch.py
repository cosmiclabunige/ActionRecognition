try:
    import tensorflow as tf
except Exception as e:
    print('Error loading modules in cnn_arch.py: ', e)


def deep_network_test_and_tflite(CNNModel, inputSize, LSTMNeurons, DenseNeurons, nClasses):
    """This function is used to encapsulate the CNN model in the time distributed layer, and append the LSTM and Dense layers to the TDL layer.
    Based on the model chosen during the class initialization, the proper CNN will be selected"""
    featuresExt = None
    if CNNModel == 1:  # proposed Model1 architecture
        featuresExt = CNN1(inputSize[1:])
    elif CNNModel == 2:  # proposed Model2 architecture
        featuresExt = CNN2(inputSize[1:])
    elif CNNModel == 3:  # proposed Model3 architecture
        featuresExt = CNN3(inputSize[1:])
    elif CNNModel == 4 or CNNModel == 5:  # MobileNetV2 for transfer learning
        featuresExt = MobileNetV2(inputSize[1:])
    elif CNNModel == 6 or CNNModel == 7:  # Xception for transfer learning
        featuresExt = Xception(inputSize[1:])
    elif CNNModel == 8 or CNNModel == 9:  # ResNet50 for transfer learning
        featuresExt = ResNet50V2(inputSize[1:])
    input_shape = tf.keras.layers.Input(inputSize)
    TD = tf.keras.layers.TimeDistributed(featuresExt)(
        input_shape)  # encapsulating the CNN model in the TDL layer
    RNN = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(
        LSTMNeurons))(TD)  # adding the LSTM layer
    Dense1 = tf.keras.layers.Dense(DenseNeurons, activation='relu')(
        RNN)  # adding the Dense layer
    Dense2 = tf.keras.layers.Dense(nClasses, activation='softmax')(
        Dense1)  # last layer performs the classification of the input
    model_ = tf.keras.models.Model(inputs=input_shape, outputs=Dense2)
    # model_.summary()
    return model_


def deep_network_train(CNNModel, inputSize, LSTMNeurons, DenseNeurons, nClasses):
    """This function is used to encapsulate the CNN model in the time distributed layer, and append the LSTM and Dense layers to the TDL layer.
    Based on the model chosen during the class initialization, the proper CNN will be selected"""
    featuresExt = None
    if CNNModel == 1:  # proposed Model1 architecture
        featuresExt = CNN1(inputSize[1:])
    elif CNNModel == 2:  # proposed Model2 architecture
        featuresExt = CNN2(inputSize[1:])
    elif CNNModel == 3:  # proposed Model3 architecture
        featuresExt = CNN3(inputSize[1:])
    elif CNNModel == 4:  # MobileNetV2 for transfer learning
        featuresExt = MobileNetV2(inputSize[1:])
    elif CNNModel == 5:  # MobileNetV2 for fine-tuning
        featuresExt = MobileNetV2FT(inputSize[1:])
    elif CNNModel == 6:  # Xception for transfer learning
        featuresExt = Xception(inputSize[1:])
    elif CNNModel == 7:  # Xception for fine-tuning
        featuresExt = XceptionFT(inputSize[1:])
    elif CNNModel == 8:  # ResNet50 for transfer learning
        featuresExt = ResNet50V2(inputSize[1:])
    elif CNNModel == 9:  # ResNet50 for fine-tuning
        featuresExt = ResNet50V2FT(inputSize[1:])
    input_shape = tf.keras.layers.Input(inputSize)
    TD = tf.keras.layers.TimeDistributed(featuresExt)(
        input_shape)  # encapsulating the CNN model in the TDL layer
    RNN = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(
        LSTMNeurons))(TD)  # adding the LSTM layer
    Dense1 = tf.keras.layers.Dense(DenseNeurons, activation='relu')(
        RNN)  # adding the Dense layer
    Dense2 = tf.keras.layers.Dense(nClasses, activation='softmax')(
        Dense1)  # last layer performs the classification of the input
    model_ = tf.keras.models.Model(inputs=input_shape, outputs=Dense2)
    model_.summary()
    return model_


# Lightest model: 5 blocks of convolutional blocks with batch normalization and average pooling layers.
def CNN1(shape):
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model


# Default model: 6 blocks of convolutional blocks with batch normalization and average pooling layers.
def CNN2(shape):
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x2 = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x2)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model


# Heaviest model: 7 blocks of convolutional blocks with batch normalization and average pooling layers.
def CNN3(shape):
    # The input of the first three blocks are summed with the output of the blocks. At the end, the global average pooling
    # layer is used to flatten the outputs
    momentum = .8
    input_img = tf.keras.Input(shape)
    x0 = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.Conv2D(
        8, (3, 3), padding='same', activation='relu')(x0)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x0])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x1 = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x1)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x2 = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x2)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x3 = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x3)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x3])
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), input_shape=shape, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    model = tf.keras.models.Model(input_img, x)
    return model


def MobileNetV2(shape):  # MobileNetV2 for transfer learning
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = False
    return base_model


def MobileNetV2FT(shape):  # MobileNetV2 for fine-tuning, last five layers are tuned
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    return base_model


def Xception(shape):  # Xception for transfer learning
    base_model = tf.keras.applications.Xception(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = False
    return base_model


def XceptionFT(shape):  # Xception for fine-tuning, last five layers are tuned
    base_model = tf.keras.applications.Xception(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    return base_model


def ResNet50V2(shape):  # ResNet50 for transfer learning
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = False
    return base_model


def ResNet50V2FT(shape):  # ResNetV2 for fine-tuning, last five layers are tuned
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=shape, include_top=False, pooling="avg")
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    return base_model
