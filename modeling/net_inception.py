from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.compat.v1.keras import initializers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

# Modified based on application example:
# https://keras.io/applications/#inceptionv3
def inception_model(input_shape, class_names):
    model = tf.keras.models.Sequential([
        InceptionV3(include_top=False, weights=None, input_shape=input_shape),  #'imagenet'  None
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.1),
        layers.Dense(1024, activation='linear', activity_regularizer=regularizers.l1(0.0001)),
        layers.LeakyReLU(alpha=0.01),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(64, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(8, activation='softmax')
    ])

    #sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    #adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return model, "inceptionv3_model", 500