import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define la red neuronal (igual que antes)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.w1 = np.random.randn(hidden_size, input_size + 1) * 0.1
        self.w2 = np.random.randn(output_size, hidden_size + 1) * 0.1
        self.learning_rate = learning_rate
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        self.layer0 = np.hstack((x, np.ones((x.shape[0], 1))))
        self.layer1_pre = np.dot(self.layer0, self.w1.T)
        self.layer1 = self.relu(self.layer1_pre)
        self.layer1 = np.hstack((self.layer1, np.ones((self.layer1.shape[0], 1))))
        self.output = np.dot(self.layer1, self.w2.T)
        return self.output
    
    def compute_error(self, x, y):
        output = self.forward(x)
        return np.mean((y - output)**2)

# Genera datos de entrada
x = np.linspace(-2, 2, 100).reshape(-1, 1)
y = x**2

# Normaliza los datos
def normalize_data(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

x_normalized = normalize_data(x)
y_normalized = normalize_data(y)

# Crear la instancia de la red neuronal
nn = NeuralNetwork(input_size=1, hidden_size=5, output_size=1)

# Establecer un rango para los pesos de w1[0, 0] y w2[0, 0]
w1_range = np.linspace(-1, 1, 20)
w2_range = np.linspace(-1, 1, 20)

# Crear una malla para los valores de w1 y w2
error_landscape = np.zeros((len(w1_range), len(w2_range)))

# Calcular el error para cada combinación de los pesos seleccionados
for i, w1_val in enumerate(w1_range):
    for j, w2_val in enumerate(w2_range):
        nn.w1[0, 0] = w1_val  # Establecer w1[0, 0]
        nn.w2[0, 0] = w2_val  # Establecer w2[0, 0]
        
        # Calcular el error para esta combinación de pesos
        error_landscape[i, j] = nn.compute_error(x_normalized, y_normalized)

# Crear una malla para graficar
W1, W2 = np.meshgrid(w1_range, w2_range)

# Graficar el landscape de error en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, error_landscape, cmap='viridis')

# Etiquetas y título
ax.set_xlabel('w1[0, 0]')
ax.set_ylabel('w2[0, 0]')
ax.set_zlabel('Error (MSE)')
ax.set_title('Landscape del Error de la Red Neuronal')

plt.show()
