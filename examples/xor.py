from layers import FCLayer, ActivationLayer
from activation_functions import tanh, tanh_prime
from loss_functions import mse, mse_prime
from network import Network
import numpy as np


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

x_test = np.array([[[0, 0]], [[0,1]], [[1,1]], [[1,0]]])

predictions = net.predict(x_test)

print(predictions)
