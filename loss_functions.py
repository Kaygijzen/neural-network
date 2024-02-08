import numpy as np


def mean_squared_error(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))


def root_mean_squared_error(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
