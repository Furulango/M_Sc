"""
Maestría en Ciencias - 07-02-2025
Red neuronal de 2 capas usando la función de activación Sigmoide
Entrada: Dos señales, x1 y x2
Salida: Dos combinaciones distintas

Tabla de entradas y salidas esperadas:
P | x1 | x2 | y1 | y2 |
0 | 0  | 0  |  0  |  1  |
1 | 0  | 1  |  1  |  0  |
2 | 1  | 0  |  1  |  0  |
3 | 1  | 1  |  0  |  1  |

Descripción:
- y1: Patrón 0110
- y2: Patrón 1001

Estructura de la red:
- Primera capa: 3 neuronas
- Segunda capa: 2 neuronas
- Función de activación: ReLU en la capa oculta, Sigmoide en la salida
- Error: MSE (Mean Squared Error)
"""

import numpy as np
import funciones as fn
import pandas as pd
import matplotlib.pyplot as plt

# Tasa de aprendizaje
n = 0.001

# Inicialización de pesos usando He Initialization (va mejor en ReLU)
w1 = np.random.randn(5, 3) * np.sqrt(2 / 2)  # 2 entradas + bias
w2 = np.random.randn(6, 2) * np.sqrt(2 / 5)  # 3 neuronas ocultas + bias (2 salidas)

# Entradas 
x = np.array([[0,0],[0,1],[1,0],[1,1]])
x = np.hstack((x, np.ones((x.shape[0], 1))))  #Bias en columna extra

# Salidas esperadas en columnas
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

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
    #Break cuando el error es menor a  1e-4
    if np.mean(error**2) < 1e-4:
        break

# Predicciones
h1 = fn.relu(np.dot(x, w1.T))
h1 = np.hstack((h1, np.ones((h1.shape[0], 1))))
h2 = fn.sigmoide(np.dot(h1, w2))

# Crear DataFrame para mostrar resultados
resultados = pd.DataFrame({
    'x1': x[:, 0],
    'x2': x[:, 1],
    'Salida Esperada Y1': y[:, 0],
    'Salida Esperada Y2': y[:, 1],
    'Salida Obtenida Y1': np.round(h2[:, 0]).flatten(),
    'Salida Obtenida Y2': np.round(h2[:, 1]).flatten()
})

print(resultados)

# Graficar el error
plt.plot(iteraciones, errores)
plt.xlabel('Iteraciones')
plt.ylabel('Error MSE')
plt.title('MSE')
plt.show()

# Guardar pesos
np.save('w1_Red_SalidaExtra.npy', w1)
np.save('w2_Red_SalidaExtra.npy', w2)
