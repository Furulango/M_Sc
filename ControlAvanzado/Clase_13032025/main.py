import numpy as np

data = np.loadtxt('data.txt', delimiter='\t', skiprows=1)  # Ajusta si el archivo tiene diferente delimitador
time = data[:, 0]
input_signal = data[:, 1]
output_signal = data[:, 2]

