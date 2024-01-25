import numpy as np
import os
from PIL import Image

# class for base neuron
class BaseNeuron:
    def __init__(self):
        pass

    def activate(self, inputs):
        raise NotImplementedError("Must be implemented by subclass.")

# class for the perceptron
class Perceptron(BaseNeuron):
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias
    
    def activate(self, inputs):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-inputs))

    def predict(self, inputs):
        output = np.dot(inputs, self.weights) + self.bias
        return self.activate(output)
    
    def update_weights(self, inputs, error, learning_rate):
        # Implementation of the weight update logic
        self.weights -= learning_rate * error * inputs
        self.bias -= learning_rate * error

    def calculate_loss(self, prediction, actual):
        # Binary cross-entropy loss function
        epsilon = 1e-7
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return - (actual * np.log(prediction) + (1 - actual) * np.log(1 - prediction))

# class for base activation function
class ActivationFunction:
    def __call__(self, x):
        raise NotImplementedError("Must be implemented by subclass.")

# class for Sigmoid activation function
class SigmoidActivationFunction(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
# class for input data
class InputData:
    def __init__(self, directory, image_size = (20, 20)):
        self.directory = directory
        self.image_size = image_size

    def load_images(self):
        images = []
        labels = []

        # Iterate over each file in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(".png"): 
                # Load the image
                img = Image.open(os.path.join(self.directory, filename))

                # Convert the image to a numpy array and normalize
                img_array = np.asarray(img) / 255.0

                # Flatten the image
                img_array = img_array.flatten()

                # Extract label from filename. The format is 'label_index.png'
                label = int(filename.split('_')[0])

                images.append(img_array)
                labels.append(label)

        return np.array(images), np.array(labels)
    
# class for training
class Training:
    def __init__(self, perceptron, learning_rate):
        self.perceptron = perceptron
        self.learning_rate = learning_rate

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                prediction = self.perceptron.predict(x)
                error = np.abs(prediction - y)
                # print('error:', error)
                self.perceptron.update_weights(x, error, self.learning_rate)
                total_loss += self.perceptron.calculate_loss(prediction, y)
            average_loss = total_loss / len(x_train)
            print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}")

# class for testing
class Testing:
    def __init__(self, perceptron):
        self.perceptron = perceptron

    def test(self, x_test):
        predictions = [self.perceptron.predict(x) for x in x_test]
        return predictions

# evaluate the accuracy
def evaluate_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    total = len(labels)
    accuracy = correct / total
    return accuracy

# main function
def main():
    # Initialize perceptron with random weights and bias
    input_size = 400  # For flattened 20x20 images
    weights = np.random.randn(input_size)
    bias = np.random.randn()
    perceptron = Perceptron(weights, bias)

    # Load training data
    data_directory = "figures"
    input_data = InputData(data_directory)
    x_train, y_train = input_data.load_images()

    # Train the perceptron
    trainer = Training(perceptron, learning_rate=0.1)
    trainer.train(x_train, y_train, epochs=100)

    test_data_directory = "test"
    test_data = InputData(test_data_directory)
    x_test, y_test = test_data.load_images()
    tester = Testing(perceptron)
    predictions = tester.test(x_test)

    # evaluate the predictions
    predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
    accuracy = evaluate_accuracy(predicted_labels, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
