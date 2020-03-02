#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os, sys
import argparse
import numpy as np
import uuid
import pandas as pd
import math
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from modelingutils import *
#from bayes_opt import BayesianOptimization
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.compat.v1.keras import initializers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pathlib

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

def convolutional_model(input_shape, class_names):
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.01), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        #layers.Dense(1024, activation=layers.LeakyReLU(alpha=0.01)), #activation='linear', activity_regularizer=regularizers.l1(0.0001)),
        layers.Dense(128, activation='linear', activity_regularizer=regularizers.l1(0.0001)),
        layers.LeakyReLU(alpha=0.01),
        layers.Dropout(0.3),
        layers.Dense(64, activation='linear', activity_regularizer=regularizers.l1(0.0001)),
        layers.LeakyReLU(alpha=0.01), 
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

    return model, "conv_model", 200

def inception_model(input_shape, class_names):
    model = tf.keras.models.Sequential([
        InceptionV3(include_top=False, weights=None, input_shape=input_shape),  #'imagenet'  None
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'), 
        layers.Dropout(0.1),
        layers.Dense(8, activation='softmax')
    ])

    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    #adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    return model, "inceptionv3_model", 200

def load_model(model_name):
    return tf.keras.models.load_model(model_name,
            custom_objects={'LeakyReLU': layers.LeakyReLU}
        ), model_name, 0

def main(p):
    data_dir = pathlib.Path(p + "Kather_texture_2016_image_tiles_5000", 'Kather_texture_2016_image_tiles_5000')
    os.listdir(data_dir)

    image_count = len(list(data_dir.glob('*/*.tif')))
    print(image_count)

    # Classification labels
    class_names = np.array([item.name for item in data_dir.glob('*') if item.name not in [".DS_Store"]])  # "08_EMPTY"
    print(class_names)

    # Data generators and augmentation
    batch_size = 128
    img_height = 150
    img_width = 150
    input_shape = (img_height, img_width, 3)

    # Data augmentation, to increase the generalizability of our classifier, we may first randomly jitter points along the distribution by adding some random values
    # In-place data augmentation or on-the-fly data augmentation https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
    # Standardization is a data scaling technique that assumes that the distribution of the data is Gaussian and shifts the distribution of the data to have a mean of zero and a standard deviation of one.
    #train_datagen = ImageDataGenerator( 
    #    rescale=1./255,
        #rotation_range=45, 
        #width_shift_range=.15,
        #height_shift_range=.15,
        #horizontal_flip=True,
        ##shear_range=0.2,
        #zoom_range=0.5,
    #    validation_split=0.2
    #)
    datagen = ImageDataGenerator(  # validation_datagen
        #rescale=1./255, # normalize
        samplewise_center=True, # std & scale
        samplewise_std_normalization=True,
        validation_split=0.1
    )
    test_datagen = ImageDataGenerator(
        #rescale=1./255, # normalize
        samplewise_center=True, # std & scale
        samplewise_std_normalization=True
    )

    # Make df of all images and split on train / test set
    traindf, testdf = read_split_imagefiles(data_dir, class_names)

    train_generator = datagen.flow_from_dataframe( 
        dataframe=traindf,
        directory=data_dir,
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset='training', 
        class_mode='categorical',
        classes = list(class_names))

    validation_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=data_dir,
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset='validation', 
        class_mode='categorical',
        classes = list(class_names))

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=data_dir,
        shuffle=False,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        classes = list(class_names))

    # Featurewise statistics
    #image_batch, label_batch = next(train_generator)
    #datagen.fit(image_batch)
    #test_datagen.fit(image_batch)

    # Inspect a batch
    #image_batch, label_batch = next(train_generator)
    #show_batch(image_batch, label_batch, class_names)

    print("Training samples: ", train_generator.samples)
    print("Validation samples: ", validation_generator.samples)
    print("Test samples: ", test_generator.samples)

    # Train
    #model, model_name, epochs = programtest_model(input_shape, class_names)
    #model, model_name, epochs = inception_model(input_shape, class_names)
    model, model_name, epochs = convolutional_model(input_shape, class_names)
    #model, model_name, epochs = biopixel_model(input_shape, class_names)
    #model, model_name, epochs = load_model("conv_model-100.h5")
    model.summary()

    #return model, test_generator, class_names

    # Setup callbacks
    #es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100)
    tb = TensorBoard(log_dir='./logs', write_graph=True, write_images=True, update_freq='epoch')
    callbacks=[tb] #,es]

    print("Train for epochs: ", epochs)

    H = model.fit(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        callbacks=callbacks
    )
        
    model.summary()
    plot_history(H)
    model.save("{}-{}.h5".format(model_name, epochs))
    #model.save("./colorectal-histology-mnist/models/{}.h5".format(model_name))

    ## Test model
    test_model(model, test_generator, class_names)

    #https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
    #https://github.com/anktplwl91/visualizing_convnets/blob/master/model_training_and_visualizations.py

    return model, test_generator, class_names

# Run as
# floyd run
if __name__ == "__main__":
    # Import images
    p = '/mnist/' #floydhub
    #p = '../../colorectal-histology-mnist/' #local
    #p = './colorectal-histology-mnist/data/' #cloud
    model, test_generator, class_names = main(p)

    #Tasks: inspect a batch, train a model, test a model, apply a model (class activation)





# %%
