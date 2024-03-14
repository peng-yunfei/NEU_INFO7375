from typing import Any
import numpy as np

class MiniBatchGenerator:
    def __init__(self, X, Y, mini_batch_size=64):
        self.X = X
        self.Y = Y
        self.mini_batch_size = mini_batch_size

    def shuffle_and_partition(self):
        m = self.X.shape[1] 
        mini_batches = []
        
        # Shuffle
        permutation = np.random.permutation(m)
        shuffled_X = self.X[:, permutation]
        shuffled_Y = self.Y[:, permutation]

        num_complete_minibatches = m // self.mini_batch_size
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*self.mini_batch_size : (k+1)*self.mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*self.mini_batch_size : (k+1)*self.mini_batch_size]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        # Handling the end case
        if m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches*self.mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*self.mini_batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        
        return mini_batches

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
   
    # softmax derivative
    def softmax_derivative(output):
        return output * (1 - output)  

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

    def train(self, X_train, Y_train, epochs, mini_batch_size):
        mini_batch_generator = MiniBatchGenerator(X_train, Y_train, mini_batch_size)
        
        for epoch in range(epochs):
            mini_batches = mini_batch_generator.shuffle_and_partition()
            epoch_cost = 0.0
            
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch

                predicted = self.forward(mini_batch_X)

                mini_batch_cost = self.loss(predicted, mini_batch_Y)
                epoch_cost += mini_batch_cost
                
                self.backpropagation(mini_batch_X, mini_batch_Y)
                
            print(f"Cost after epoch {epoch}: {epoch_cost / len(mini_batches)}")
    
