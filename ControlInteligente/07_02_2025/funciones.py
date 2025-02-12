
import numpy as np

def y_xx(x):
    return 2*x

def sigmoide(x):
    return 1/(1+np.exp(-x))

def sigmoide_derivada(x):
    return sigmoide(x)*(1-sigmoide(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return np.where(x > 0, 1, 0)
