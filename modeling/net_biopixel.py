from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.compat.v1.keras import initializers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler

#Images are:
# * Patterns, not objects -> we should recognise patterns
#   Vertical, horizontal lines
#   Points, bullets
#   Colors
# * We need to build deep networks, but
#   Bigger the model, more prone it is to overfitting
#   “neurons that fire together, wire together
# https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/

def biopixel_model(input_shape, class_names):

    classes = len(class_names)
    kernel_init = initializers.glorot_uniform()
    bias_init = initializers.Constant(value=0.2)
    
    def identity_block(X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3, F4 = filters

        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X1 = layers.Conv2D(filters=F1, kernel_size=(3, 9), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=kernel_init)(X)
        X1 = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X1)
        X1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(X1)
        X1 = layers.Activation('relu')(X1)

        # Second component of main path
        X2 = layers.Conv2D(filters=F2, kernel_size=(9, 3), strides=(1, 1), padding='valid', name=conv_name_base + '2b', kernel_initializer=kernel_init)(X)
        X2 = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X2)
        X2 = layers.MaxPooling2D((3, 3), strides=(2, 2))(X2)
        X2 = layers.Activation('relu')(X2)

        # Third component of main path
        X3 = layers.Conv2D(filters=F3, kernel_size=(9, 9), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=kernel_init)(X)
        X3 = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X3)
        X3 = layers.MaxPooling2D((3, 3), strides=(2, 2))(X3)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='valid')(X)
        X4 = layers.Conv2D(filters=F4, kernel_size=(1, 1), padding='valid', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(X4)

        print(X1.shape)
        print(X2.shape)
        print(X3.shape)
        print(X4.shape)

        X = layers.concatenate([X1, X2, X3, X4], axis=4)
        
        #X = layers.Add()([X1, X2, X3, X4])
        #X = layers.Activation('relu')(X)

        return X

    def convolutional_block(X, f, filters, stage, block, s=1):

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1 = filters

        X = layers.Conv2D(256, (f, f), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=kernel_init)(X)
        X = layers.Conv2D(256, (f, f), strides=(s, s), name=conv_name_base + '2b', kernel_initializer=kernel_init)(X)
        X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Final step
        X = layers.Activation('relu')(X)

        return X

    # Define the input as a tensor with shape input_shape
    X_input = layers.Input(input_shape)

    # Zero-Padding
    #X = layers.ZeroPadding2D((3, 3))(X_input)
    X = X_input

    # Stage 1
    #X = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=initializers.glorot_uniform(seed=0))(X)
    #X = layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    #X = layers.Activation('relu')(X)
    #X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    #X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 64, 64], stage=2, block='b')
    #X = identity_block(X, 3, [64, 64, 64], stage=2, block='c')
    X = convolutional_block(X, f=3, filters=[128], stage=2, block='a', s=1)
    

    ### START CODE HERE ###
    
    # Stage 3
    #X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    #X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    #X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    #X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    #X = layers.AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = layers.Flatten()(X)
    #X = layers.Dense(classes*128, activation=layers.LeakyReLU(alpha=0.01))(X)
    #X = layers.Dense(classes*128, activation=layers.LeakyReLU(alpha=0.01))(X)
    X = layers.Dense(classes*8, activation=layers.LeakyReLU(alpha=0.01))(X)
    X = layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = models.Model(inputs=X_input, outputs=X, name='biopixel_v1')

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model, "biopixel_model", 10