import numpy as np

class InputNormalization:
    def __init__(self, data):
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.std = np.std(data, axis=1, keepdims=True)

    def normalize(self, data):
        normalized_data = (data - self.mean) / self.std
        return normalized_data

data = np.array([[1, 8, 3, 9, 5],
                 [5, 1, 0, 2, 1],
                 [2, 3, 9, 10, 6]])

normalizer = InputNormalization(data)
print("Original Data:\n", data)
print("Normalized Data:\n", normalizer.normalize(data))

