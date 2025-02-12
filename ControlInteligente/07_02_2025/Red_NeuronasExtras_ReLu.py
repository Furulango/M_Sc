"""
Maestria en ciencias - 07-02-2025
Red neuronal 2 capas usando la funcion de activacion Sigmoide
Entrada dos senales, x1 y x2 
Salida sera una combinacion XOR
P | x1 | x2 | y |
0 | 0  | 0  | 0 |
1 | 0  | 1  | 1 |
2 | 1  | 0  | 1 |
3 | 1  | 1  | 0 |

Primera capa: 3 neuronas
Segunda capa: 1 neurona
Funcion de activacion: Sigmoide
Error: MSE
"""

import numpy as np
import funciones as fn
import pandas as pd
import matplotlib.pyplot as plt

# Tasa de aprendizaje
n = 0.01

# Inicialización de pesos usando He Initialization (va mejor en ReLU)
w1 = np.random.randn(3, 3) * np.sqrt(2 / 2)  # 2 entradas + bias
w2 = np.random.randn(4, 1) * np.sqrt(2 / 3)  # 3 neuronas ocultas + bias

# Entradas 
x = np.array([[0,0],[0,1],[1,0],[1,1]])
x = np.hstack((x, np.ones((x.shape[0], 1))))  #Bias en columna extra

# Salidas esperadas en columnas
y = np.array([[0], [1], [1], [0]])

# Almacenamiento de las iteraciones y el MSE para graficar
iteraciones = []
errores = []

# Entrenamiento
for i in range(10000000):
    # Propagación hacia adelante
    h1 = fn.relu(np.dot(x, w1.T)) 
    h1 = np.hstack((h1, np.ones((h1.shape[0], 1))))  
    h2 = fn.sigmoide(np.dot(h1, w2))  

    # Calculo del error
    error = y - h2
    if i % 1000 == 0:
        print(f"Iteración {i}, Error MSE: {np.mean(error**2)}")
        #Almacenamiento de las iteraciones y el MSE para graficar
        iteraciones.append(i)
        errores.append(np.mean(error**2))

    # Retropropagación
    d_h2 = error * fn.sigmoide_derivada(h2) 
    d_w2 = np.dot(h1.T, d_h2)  

    d_h1 = np.dot(d_h2, w2.T) * fn.relu_derivada(h1)
    d_w1 = np.dot(x.T, d_h1[:, :-1])  

    # Actualización de pesos
    w2 += n * d_w2
    w1 += n * d_w1.T  
    #Break cuando el error es menor a 0.0001
    if np.mean(error**2) < 0.000001:
        break

# Predicciones
h1 = fn.relu(np.dot(x, w1.T))
h1 = np.hstack((h1, np.ones((h1.shape[0], 1))))
h2 = fn.sigmoide(np.dot(h1, w2))

# Crear DataFrame para mostrar resultados
resultados = pd.DataFrame({
    'x1': x[:, 0],
    'x2': x[:, 1],
    'Salida Esperada': y.flatten(),
    'Salida Obtenida': np.round(h2).flatten()
})

print(resultados)

# Graficar el error
plt.plot(iteraciones, errores)
plt.xlabel('Iteraciones')
plt.ylabel('Error MSE')
plt.title('MSE')
plt.show()

# Guardar pesos
np.save('w1_Red_NeuronasExtras.npy', w1)
np.save('w2_Red_NeuronasExtras.npy', w2)


