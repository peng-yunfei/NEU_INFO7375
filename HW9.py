from typing import Any
import numpy as np

class Activation:
    # Develop linear, ReLU, sigmoid, tanh, and softmax activation functions       
    def linear(input):
        return input
    
    def ReLU(input):
        return max(0, input)
    
    def sigmoid(input):
        return 1 / (1 + np.exp(-input))
    
    def tanh(input):
        return np.tanh(input)
    
    def softmax(input):
        return np.exp(input) / sum(np.exp(input))

class Layer:
    # add dropout rate
    def __init__(self, input_size, output_size, activation, dropout_rate=0.0):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn((output_size, 1))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.output = None

    def forward(self, input, training=True):
        z = np.dot(self.weights, input) + self.bias
        self.output = self.activation(z)

        # drop out
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, self.output.shape)
            self.output = np.multiply(self.output, self.dropout_mask) / (1 - self.dropout_rate)
        
        return self.output

class DeepNeuralNetwork:
    def __init__(self, input_size, output_size, l2_reg_lambda=0.0):
        self.layers = []
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.l2_reg_lambda = l2_reg_lambda

    def add_layer(self, layer):
        self.layers.append(layer)
        return self.layers

    def forward(self, input, training=True):
        for layer in self.layers:
            input = layer.forward(input, training)
        self.output = input
        return self.output
    
    def compute_loss(self, predicted, actual):
        m = actual.shape[1]
        data_loss = -1/m * np.sum(actual * np.log(predicted + 1e-15))
        reg_loss = 0.5 * self.l2_reg_lambda * sum(np.linalg.norm(layer.weights) ** 2 for layer in self.layers)
        return np.squeeze(data_loss + reg_loss)

    def backpropagation(self, input, actual):
        predicted = self.forward(input)
        m = actual.shape[1]

        loss = self.compute_loss(predicted, actual)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == 0:
                layer_input = input
            else:
                layer_input = self.layers[i - 1].output

            # Sigmoid activation
            derivative_activation = layer.output * (1 - layer.output)

            # dropout
            if layer.dropout_rate > 0.0:
                derivative_activation = np.multiply(derivative_activation, layer.dropout_mask) / (1 - layer.dropout_rate)

            loss *= derivative_activation
            d_weights = np.dot(layer_input, loss.T) / m
            d_bias = np.sum(loss, axis=1, keepdims=True) / m

            # L2 regularization
            d_weights += (self.l2_reg_lambda * layer.weights) / m

            layer.weights -= d_weights.T
            layer.bias -= d_bias

            if i > 0:
                prev_layer = self.layers[i - 1]
                loss = np.dot(prev_layer.weights.T, loss)
