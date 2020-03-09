from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.compat.v1.keras import initializers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

def programtest_model(input_shape, class_names):
    model = tf.keras.models.Sequential([
        layers.Conv2D(2, (3, 3), activation=layers.LeakyReLU(alpha=0.1), input_shape=input_shape),
        layers.Flatten(),
        layers.Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model, "test_model", 2

#GPU with conv_model and Data Augmentation batchnorm
#Nå: GPU with conv_model and Data Augmentation difference filters
#GPU with conv_model and Data Augmentation batchnorm, differnce filters

def convolutional_model(input_shape, class_names):
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.01), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        #layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        #layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        #layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        #layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(1024, activation='linear', activity_regularizer=regularizers.l1(0.0001)),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(8, activation='softmax')
    ])

    # https://keras.io/optimizers/
    #decay_rate = 0.01 / 100. #learning_rate / epochs
    #sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False) #, decay=decay_rate)
    #rmsprop = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    #adagrad = optimizers.Adagrad(learning_rate=0.01)
    #adadelta = optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    #adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model, "conv_model", 500