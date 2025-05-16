
import numpy as np

#Funciones para una red neuronal
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)