import numpy as np

class Activation:
    @staticmethod
    def ReLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    # Derivatives
    @staticmethod
    def ReLU_deriv(Z):
        return Z > 0

    @staticmethod
    def sigmoid_deriv(A):
        return A * (1 - A)

class Layer:
    def __init__(self, input_dim, output_dim, activation_func, activation_deriv):
        self.weights = np.random.randn(output_dim, input_dim) * 0.1
        self.bias = np.zeros((output_dim, 1))
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.Z = None  # Pre-activation parameter
        self.A = None  # Post-activation parameter
        self.A_prev = None  # Input to this layer

    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.weights, A_prev) + self.bias
        self.A = self.activation_func(self.Z)
        return self.A

    def backward(self, dA, learning_rate):
        m = self.A_prev.shape[1]
        # Correctly use the derivative of the activation function
        if self.activation_deriv == Activation.sigmoid_deriv:
            dZ = dA * self.activation_deriv(self.A)  
        else:
            dZ = dA * self.activation_deriv(self.Z)  #
        
        dW = np.dot(dZ, self.A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.weights.T, dZ)
        
        # Update weights and biases
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        
        return dA_prev


class DeepNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.losses = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[1]
        loss = -1/m * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return loss

    def backward(self, Y, Y_hat, learning_rate):
        dA = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.losses.append(loss)
            self.backward(Y, Y_hat, learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, loss: {loss}')


# Define the network
nn = DeepNeuralNetwork()
nn.add_layer(Layer(2, 10, Activation.ReLU, Activation.ReLU_deriv))
nn.add_layer(Layer(10, 8, Activation.ReLU, Activation.ReLU_deriv))
nn.add_layer(Layer(8, 8, Activation.ReLU, Activation.ReLU_deriv))
nn.add_layer(Layer(8, 4, Activation.ReLU, Activation.ReLU_deriv))
nn.add_layer(Layer(4, 1, Activation.sigmoid, Activation.sigmoid_deriv))

# 2 features, 1000 examples
X = np.random.randn(2, 1000)  
Y = (np.sum(X**2, axis=0) < 1).reshape(1, 1000) 

# Train the network
nn.train(X, Y, epochs=2000, learning_rate=0.01)

