import numpy as np

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, lr):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, lr):
        # output_error (dBias) dE/dB = dE/dY
        input_error = np.dot(output_error, self.weights.T) # dE/dX = dE/dY * W^T
        weights_error = np.dot(self.input.T, output_error) # dE/dW = x^T * dE/dY

        self.weights -= lr * weights_error
        self.bias -= lr * output_error
        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, output_error, lr):
        # note: no learnable params, so lr not used
        return self.activation_prime(self.output) * output_error
