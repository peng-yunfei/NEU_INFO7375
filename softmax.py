import numpy as np

class Softmax:
    def __call__(self, logits):
        """
        Parameters:
        - logits: A numpy array of numbers.

        Returns:
        - A numpy array of softmax results
        """
        exp_logits = np.exp(logits)
        sum_exp_logits = np.sum(exp_logits)
        probabilities = exp_logits / sum_exp_logits
        return probabilities

if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = Softmax()(logits)
    print(probabilities)
