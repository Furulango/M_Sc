"""
Maestria Ciencias 11-02-2025
Red Neuronal Con Backpropagation 
Funcion a entrenenar: y = x^2
Dos entradas y una salida



"""
import numpy as np
import funciones as fn
import pandas as pd
import matplotlib.pyplot as plt

# Entrada 1 señal para la función x^2
x = np.linspace(-5, 5, 100).reshape(-1, 1)
x = np.hstack((x, np.ones((x.shape[0], 1))))  # Bias en columna extra

# Salidas esperadas en columnas (función a entrenar: y = x^2)
y = x[:, 0] ** 2
y = y.reshape(-1, 1)

# Tasa de aprendizaje
n = 0.001

# Inicialización de pesos usando He Initialization (mejor para ReLU)
w1 = np.random.randn(5, 2) * np.sqrt(2 / 2)  # 1 entrada + bias para 5 neuronas en capa oculta
w2 = np.random.randn(1, 6) * np.sqrt(2 / 5)  # 5 neuronas en capa oculta + bias para 1 salida

# Almacenamiento de las iteraciones y el MSE para graficar
iteraciones = []
errores = []

# Entrenamiento
for i in range(10000000):
   # Propagación hacia adelante
    h1 = fn.relu(np.dot(x, w1.T))  # ReLU en capa oculta
    h1 = np.hstack((h1, np.ones((h1.shape[0], 1))))  # Bias en capa oculta
    h2 = np.dot(h1, w2.T)  # Capa de salida sin activación
    # Cálculo del error

    error = y - h2
    if i % 1000 == 0:
        print(f"Iteración {i}, Error MSE: {np.mean(error**2)}")
        # Almacenamiento de las iteraciones y el MSE para graficar
        iteraciones.append(i)
        errores.append(np.mean(error**2))
        
    # Retropropagación
    d_h2 = error  # No derivada de sigmoide porque no hay activación en la capa de salida
    d_w2 = np.dot(h1.T, d_h2)  # Ajustar d_w2 para que sea de tamaño (6,1)

    # Retropropagación de la capa oculta
    d_h1 = np.dot(d_h2, w2) * fn.relu_derivada(h1)
    d_w1 = np.dot(x.T, d_h1[:, :-1])  # Excluir el sesgo

    # Actualización de pesos
    w2 += n * d_w2.T  # Asegúrate de que d_w2.T tenga la forma correcta
    w1 += n * d_w1.T  

    # Salir cuando el error sea suficientemente pequeño
    if np.mean(error**2) < 0.000001:
        break


# Predicciones
h1 = fn.relu(np.dot(x, w1.T))
h1 = np.hstack((h1, np.ones((h1.shape[0], 1))))  # Bias en capa oculta
h2 = np.dot(h1, w2.T)  # Capa de salida sin función de activación

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
np.save('w1_Red_NeuronasOcultas.npy', w1)
np.save('w2_Red_NeuronasOcultas.npy', w2)
