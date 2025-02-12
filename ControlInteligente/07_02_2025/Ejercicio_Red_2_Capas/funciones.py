
import numpy as np

def sigmoide(x):
    return 1/(1+np.exp(-x))

def sigmoide_derivada(x):
    return sigmoide(x)*(1-sigmoide(x))