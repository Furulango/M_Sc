"""
Maestria en ciencias - 24-01-2025
Algoritmo de retropropagación para un perceptrón simple
Este programa implementa un perceptrón simple para resolver el problema lógico OR.
El perceptrón se entrena utilizando el algoritmo de retropropagación.
La función de activación utilizada es el límite duro (Hard limit).
Variables:
    X (numpy.ndarray): Matriz de entrada con las combinaciones de valores para el problema OR.
    y (numpy.ndarray): Vector de salida deseada para cada combinación de entrada.
    n (int): Tasa de aprendizaje.
    w1 (float): Peso sináptico inicial para la primera entrada.
    w2 (float): Peso sináptico inicial para la segunda entrada.
    w3 (float): Peso sináptico inicial para el sesgo (bias).
    data (list): Lista para almacenar los datos de cada iteración del entrenamiento.
    epochs (int): Número de épocas para el entrenamiento.
    -- for --
    epoch (int): Contador de épocas.
    j (int): Índice para iterar sobre las combinaciones de entrada.
    s (float): Suma ponderada de las entradas y el sesgo.
    h (int): Salida del perceptrón después de aplicar la función de activación.
    error (int): Diferencia entre la salida deseada y la salida del perceptrón.
    df (pandas.DataFrame): DataFrame para almacenar y mostrar los resultados del entrenamiento.
"""

import numpy as np
import pandas as pd

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
n = 1
w1 = 1
w2 = 1
w3 = -1

data = []
epochs = 10000

for epoch in range(epochs):
    for j in range(4):
        s = X[j][0]*w1 + X[j][1]*w2 + w3
        # Funcion de activacion Hard limit
        if s >= 0:
            h = 1
        else:
            h = 0
        error = y[j] - h
        w1 = w1 + error*n*X[j][0]
        w2 = w2 + error*n*X[j][1]
        w3 = w3 + error
        data.append([epoch + 1, X[j][0], X[j][1], y[j], s, h, error, w1, w2, w3])

df = pd.DataFrame(data, columns=['epoch', 'x1', 'x2','y', 's', 'h', 'e', 'w1', 'w2', 'w3'])
print(df)
