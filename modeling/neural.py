#%%
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
from bayes_opt import BayesianOptimization

#Dropout references
#https://arxiv.org/abs/1207.0580 (2012)
#http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf (2014)

# Add Dropout layer, base class

class DenseLayer():
    def __init__(self, neurons, inputs, dropout=0., bias=0.01):
        self.neurons = neurons
        self.inputs = inputs
        self.synaptic_weights = 2 * np.random.random((inputs, neurons)) - 1  #between -1 and 1
        # self.weights = np.random.normal(loc=0.0, 
                                        #scale = np.sqrt(2/(input_units+output_units)), 
                                        #size = (input_units,output_units))
        self.bias = np.full(neurons, bias)   #http://cs231n.github.io/neural-networks-2/ init of bias

class NeuralNetwork():
    def __init__(self, layers, learning_rate=0.1):
        self.layers = layers
        self.loss = []
        self.learning_rate = learning_rate

        # initialisering network activation for deep layers
        self.nonlinearity_deep = self.__leaky_relU
        self.nonlinearity_deep_prime = self.__leaky_relU_prime

        # initialising network activation function for output layer
        self.nonlinearity_out = self.__sigmoid
        self.nonlinearity_out_prime = self.__sigmoid_prime

        # initialising the cost function to use
        self.costfunction = self.__delta_cost 

    def __sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def __sigmoid_prime(self, x):
        # Good source for deriving this prime: https://beckernick.github.io/sigmoid-derivative-neural-network/ 
        return self.__sigmoid(x) * (1. - self.__sigmoid(x))

    def __relU(self, x):
        return x * (x > 0)  # faster than np.maximum(x, 0., x)

    def __relU_prime(self, x):
        # Deriving relu: https://medium.com/@yashgarg1232/derivative-of-neural-activation-function-64e9e825b67
        return 1. * (x >= 0.)

    def __leaky_relU(self, x):
        return np.where(x > 0, x, x * 0.01)  #np.maximum(x, 0.01*x, x)

    def __leaky_relU_prime(self, x):
        return np.where(x < 0., 0.01, 1.)

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_prime(self, x):
        return 1. - (self.__tanh(x) ** 2.)

    def __linear(self, x):
        return x

    def __linear_prime(self, x):
        return 1

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def __normalize(self, x):
        return x / np.max(x)

    def __dropout(self, probability):
        return np.random.random() < probability

    def __delta_cost(self, y, o):
        return y-o

    def __quadratic_cost(self, y, o):
        return np.square(y-o) / 2

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, x_train, y_train, epochs):

        for iteration in range(epochs):
            # Pass the training set through our neural network
            y_pred = self.forward(x_train) 
            
            # Backpropagate
            for i in reversed(range(len(self.layers))):
                # Calculate the error & error weighted derivative
                if i is len(self.layers)-1: #output layer
                    self.layers[i].error = self.costfunction(y_train, self.layers[i].output)
                    self.layers[i].delta = self.layers[i].error * self.nonlinearity_out_prime(self.layers[i].output) 
                else:
                    self.layers[i].error = self.layers[i+1].delta.dot(self.layers[i+1].synaptic_weights.T)
                    self.layers[i].delta = self.layers[i].error * self.nonlinearity_deep_prime(self.layers[i].output) 

                # Calculate how much to adjust the weights
                if i is 0: #input layer
                    self.layers[i].adjustment = x_train.T.dot(self.layers[i].delta) / self.layers[i].delta.shape[0]
                else:
                    self.layers[i].adjustment = self.layers[i-1].output.T.dot(self.layers[i].delta) / self.layers[i].delta.shape[0]
                
                # Adjust the weights & bias
                self.layers[i].synaptic_weights += (self.layers[i].adjustment * self.learning_rate)
                self.layers[i].bias -= (self.layers[i].delta / self.layers[i].delta.shape[0]).mean(axis=0) * self.learning_rate

            self.loss.append(np.average(np.abs(self.layers[len(self.layers)-1].error)))

    def forward(self, inputs): 
        activation = inputs
        for l in range(len(self.layers)): # Feed forward
            if l is len(self.layers)-1: #output layer
                activation = self.nonlinearity_out(np.dot(activation, self.layers[l].synaptic_weights) + self.layers[l].bias)
            else:
                activation = self.nonlinearity_deep(np.dot(activation, self.layers[l].synaptic_weights) + self.layers[l].bias)
            self.layers[l].output = activation

        return activation

    def print_weights(self):
        with np.printoptions(precision=3, suppress=True, edgeitems=30, linewidth=100000):
            for idx, layer in enumerate(self.layers):
                print("    Layer {} ({} neurons, each with {} inputs): ".format(idx, layer.synaptic_weights.shape[1], layer.synaptic_weights.shape[0]))
                print(layer.synaptic_weights)
                print("    Output")
                print(layer.output)

def neural_bayesian_optimization(x_train, x_test, y_train, y_test,onehot,class_names):
    # https://github.com/fmfn/BayesianOptimization

    # parameter grid
    param_grid = {      "learning_rate"   : ( 0.001, 0.20 ), 
                        "neurons1"        : ( 3, 40),
                        #"neurons2"        : ( 3, 40),
                        #"neurons3"        : ( 3, 40),
                        "epochs"          : ( 5000, 40000) } 

    # function to be optimized
    def neural_function(learning_rate, neurons1, epochs):
        assert type(neurons1) == int # discrete parameters!
        #assert type(neurons2) == int
        #assert type(neurons3) == int
        assert type(epochs) == int

        # Create layers 
        layers = [DenseLayer(neurons=neurons1, inputs=16)]
        #layers.append(DenseLayer(neurons=neurons2, inputs=layers[len(layers)-1].neurons))
        #layers.append(DenseLayer(neurons=neurons3, inputs=layers[len(layers)-1].neurons))
        layers.append(DenseLayer(neurons=8, inputs=layers[len(layers)-1].neurons))
        neural_network = NeuralNetwork(layers, learning_rate=learning_rate)

        # Train the neural network using the training set.
        neural_network.train(x_train, y_train, epochs)

        # Test the neural network with a new situation.
        #y_train_pred = onehot.inverse_transform(neural_network.forward(x_train)).astype(int)
        y_test_pred = onehot.inverse_transform(neural_network.forward(x_test)).astype(int)
        #y_train_true = onehot.inverse_transform(y_train).astype(int)
        y_test_true = onehot.inverse_transform(y_test).astype(int)

        return accuracy_score(y_test_true, y_test_pred)

    # wrapper function for discrete parameters
    def neural_optimizer(learning_rate, neurons1, epochs):
        neurons1 = int(neurons1)
        #neurons2 = int(neurons2)
        #neurons3 = int(neurons3)
        epochs = int(epochs)
        return neural_function(learning_rate, neurons1, epochs)

    optimizer = BayesianOptimization(
        f=neural_optimizer,
        pbounds=param_grid,
        random_state=1,
    )

    optimizer.maximize(
        n_iter=200
    )

    print(optimizer.max)

if __name__ == "__main__":
   
    np.set_printoptions(precision=3, suppress=True, edgeitems=30, linewidth=100000)
    np.random.seed(10)

    # Read data
    x,y,onehot,class_names = read_data()

    # Split data
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1001)
    x_train, x_test = x_train / np.max(x_train), x_test / np.max(x_test) # normalise
    y_train = y_train.reshape(y_train.shape[0],-1)#.astype(float)
    y_test = y_test.reshape(y_test.shape[0],-1)#.astype(float)
    
    a= False
    if a:
        neural_bayesian_optimization(x_train, x_test, y_train, y_test, onehot, class_names)
    else:
        # Create layers 
        layers = [DenseLayer(neurons=40, inputs=16)]
        layers.append(DenseLayer(neurons=40, inputs=layers[len(layers)-1].neurons))
        layers.append(DenseLayer(neurons=40, inputs=layers[len(layers)-1].neurons))
        layers.append(DenseLayer(neurons=8, inputs=layers[len(layers)-1].neurons))
        neural_network = NeuralNetwork(layers, learning_rate=0.1)

        for i in range(4):
            # Train the neural network using the training set.
            neural_network.train(x_train, y_train, 5000)

            # Test the neural network with a new situation.
            y_train_pred = onehot.inverse_transform(neural_network.forward(x_train)).astype(int)
            y_test_pred = onehot.inverse_transform(neural_network.forward(x_test)).astype(int)
            y_train_true = onehot.inverse_transform(y_train).astype(int)
            y_test_true = onehot.inverse_transform(y_test).astype(int)

            print("Train accuracy: {:.4f}".format(accuracy_score(y_train_pred, y_train_true)))   #np.corrcoef(y_train_pred.T, y_train_true.T)[0][1]))
            print("Test  accuracy: {:.4f}".format(accuracy_score(y_test_pred, y_test_true)))    #np.corrcoef(y_test_pred.T, y_test_true.T)[0][1]))
        
            print("Synaptic weights after training: ")
            #neural_network.print_weights()

            plt.plot(neural_network.loss)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()

            show_confusion(y_test_true, y_test_pred, class_names)   


