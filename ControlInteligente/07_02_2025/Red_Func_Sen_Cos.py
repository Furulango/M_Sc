import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_data(x):
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

def denormalize_data(x_norm, x_original):
    x_min = np.min(x_original)
    x_max = np.max(x_original)
    return (x_norm + 1) * (x_max - x_min) / 2 + x_min

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.w1 = np.random.randn(hidden_size, input_size + 1) * 0.1
        self.w2 = np.random.randn(output_size, hidden_size + 1) * 0.1
        self.learning_rate = learning_rate
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def clip_gradients(self, gradients, max_value=1.0):
        norm = np.linalg.norm(gradients)
        if norm > max_value:
            gradients = gradients * (max_value / norm)
        return gradients
    
    def forward(self, x):
        self.layer0 = np.hstack((x, np.ones((x.shape[0], 1))))
        self.layer1_pre = np.dot(self.layer0, self.w1.T)
        self.layer1 = self.relu(self.layer1_pre)
        self.layer1 = np.hstack((self.layer1, np.ones((self.layer1.shape[0], 1))))
        self.output = np.dot(self.layer1, self.w2.T)
        return self.output
    
    def backward(self, x, y, output):
        output_error = y - output
        d_output = output_error
        
        d_w2 = np.dot(d_output.T, self.layer1)
        d_hidden = np.dot(d_output, self.w2)
        d_hidden = d_hidden[:, :-1] * self.relu_derivative(self.layer1_pre)
        
        d_w1 = np.dot(d_hidden.T, self.layer0)
        
        d_w1 = self.clip_gradients(d_w1)
        d_w2 = self.clip_gradients(d_w2)
        
        self.w1 += self.learning_rate * d_w1
        self.w2 += self.learning_rate * d_w2
        
        return np.mean(output_error**2)

# Generar datos para seno y coseno
x = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Normalizar datos
x_normalized = normalize_data(x)
y_sin_normalized = normalize_data(y_sin)
y_cos_normalized = normalize_data(y_cos)

# Crear y entrenar red para seno
nn_sin = NeuralNetwork(input_size=1, hidden_size=20, output_size=1, learning_rate=0.001)
errors_sin = []
iterations_sin = []

# Entrenamiento seno
for i in range(50000):
    output = nn_sin.forward(x_normalized)
    error = nn_sin.backward(x_normalized, y_sin_normalized, output)
    
    if i % 1000 == 0:
        print(f"Seno - Iteración {i}, Error MSE: {error}")
        iterations_sin.append(i)
        errors_sin.append(error)
        
    if error < 0.0001:
        break

# Crear y entrenar red para coseno
nn_cos = NeuralNetwork(input_size=1, hidden_size=1000, output_size=1, learning_rate=0.001)
errors_cos = []
iterations_cos = []

# Entrenamiento coseno
for i in range(50000):
    output = nn_cos.forward(x_normalized)
    error = nn_cos.backward(x_normalized, y_cos_normalized, output)
    
    if i % 1000 == 0:
        print(f"Coseno - Iteración {i}, Error MSE: {error}")
        iterations_cos.append(i)
        errors_cos.append(error)
        
    if error < 0.0001:
        break

# Predicciones
sin_predictions = nn_sin.forward(x_normalized)
cos_predictions = nn_cos.forward(x_normalized)

denormalized_sin = denormalize_data(sin_predictions, y_sin)
denormalized_cos = denormalize_data(cos_predictions, y_cos)

# Crear DataFrame con resultados
results_sin = pd.DataFrame({
    'Input': x.flatten(),
    'Expected_Sin': y_sin.flatten(),
    'Predicted_Sin': denormalized_sin.flatten()
})

results_cos = pd.DataFrame({
    'Input': x.flatten(),
    'Expected_Cos': y_cos.flatten(),
    'Predicted_Cos': denormalized_cos.flatten()
})

print("\nResultados Seno:")
print(results_sin)
print("\nResultados Coseno:")
print(results_cos)

# Visualización
plt.figure(figsize=(15, 10))

# Gráfico de error para seno
plt.subplot(2, 2, 1)
plt.plot(iterations_sin, errors_sin)
plt.xlabel('Iteraciones')
plt.ylabel('MSE')
plt.title('Error de Entrenamiento - Seno')

# Gráfico de error para coseno
plt.subplot(2, 2, 2)
plt.plot(iterations_cos, errors_cos)
plt.xlabel('Iteraciones')
plt.ylabel('MSE')
plt.title('Error de Entrenamiento - Coseno')

# Predicciones vs real para seno
plt.subplot(2, 2, 3)
plt.scatter(x, y_sin, label='Real', alpha=0.5)
plt.scatter(x, denormalized_sin, label='Predicción', alpha=0.5)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Predicciones vs Real - Seno')
plt.legend()

# Predicciones vs real para coseno
plt.subplot(2, 2, 4)
plt.scatter(x, y_cos, label='Real', alpha=0.5)
plt.scatter(x, denormalized_cos, label='Predicción', alpha=0.5)
plt.xlabel('x')
plt.ylabel('cos(x)')
plt.title('Predicciones vs Real - Coseno')
plt.legend()

plt.tight_layout()
plt.show()

# Guardar pesos
np.save('w1_sin.npy', nn_sin.w1)
np.save('w2_sin.npy', nn_sin.w2)
np.save('w1_cos.npy', nn_cos.w1)
np.save('w2_cos.npy', nn_cos.w2)