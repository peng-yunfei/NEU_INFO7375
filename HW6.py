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
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size, 1)
        self.activation = activation
        self.output = None

    def forward(self, input):
        z = np.dot(self.weights, input) + self.bias
        self.output = self.activation(z)
        return self.output

class DeepNeuralNetwork:
    def __init__(self, input_size, output_size):
        self.layers = []
        self.output = None
        self.input_size = input_size
        self.output_size = output_size

    def add_layer(self, layer):
        self.layers.append(layer)
        return self.layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input
        return self.output
    
    def loss(self, predicted, actual):
        m = actual.shape[1]
        loss = -1/m * np.sum(actual * np.log(predicted + 1e-15))
        return np.squeeze(loss)
    
    def backpropagation(self, input, actual):
        predicted = self.forward(input)
        m = actual.shape[1]

        loss = loss(predicted, actual)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == 0:
                layer_input = input
            else:
                layer_input = self.layers[i - 1].output

            # Assuming sigmoid activation
            derivative_activation = layer.output * (1 - layer.output)

            loss *= derivative_activation
            d_weights = np.dot(layer_input, loss.T) / m
            d_bias = np.sum(loss, axis=1, keepdims=True) / m

            layer.weights -= d_weights.T
            layer.bias -= d_bias

            if i > 0:
                prev_layer = self.layers[i - 1]
                loss = np.dot(prev_layer.weights, loss)

    