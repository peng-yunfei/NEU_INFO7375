class ActivationFunction:
    def activation(sum, threshold):
        # if the weighted sum is less than threshold, return 0
        # otherwise, return 1
        return 0 if sum < threshold else 1

class Neuron:
    def __init__(self, weights, threshold):
        # initialize weights and threshold
        self.weights = weights
        self.threshold = threshold
    
    def output(self, inputs):
        # calculate the weighted sum of the inputs
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        
        # use the activation function
        return ActivationFunction.activation(sum, self.threshold)

# Example 
# initialize a neuron
weights = [0.5, -0.5, 0.3]
threshold = 0.2
neuron = Neuron(weights, threshold)

inputs = [0, 1, 0]
output = neuron.output(inputs)
print(f"Output: ", output)