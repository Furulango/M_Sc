import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

def normalize_data(x):
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

def denormalize_data(x_norm, x_original):
    x_min = np.min(x_original)
    x_max = np.max(x_original)
    return (x_norm + 1) * (x_max - x_min) / 2 + x_min

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.w1 = np.random.randn(hidden_size, input_size + 1) * 0.1  # +1 for bias
        self.w2 = np.random.randn(output_size, hidden_size + 1) * 0.1  # +1 for bias
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
        # Bias
        self.layer0 = np.hstack((x, np.ones((x.shape[0], 1))))
        
        # Capa oculta
        self.layer1_pre = np.dot(self.layer0, self.w1.T)
        self.layer1 = self.relu(self.layer1_pre)
        self.layer1 = np.hstack((self.layer1, np.ones((self.layer1.shape[0], 1))))
        
        # Salida
        self.output = np.dot(self.layer1, self.w2.T)
        return self.output
    
    def backward(self, x, y, output):
        # Error salida
        output_error = y - output
        d_output = output_error
        
        # Error capa oculta
        d_w2 = np.dot(d_output.T, self.layer1)
        d_hidden = np.dot(d_output, self.w2)
        d_hidden = d_hidden[:, :-1] * self.relu_derivative(self.layer1_pre)
        
        # Error entrada
        d_w1 = np.dot(d_hidden.T, self.layer0)
        
        # Gradiente
        d_w1 = self.clip_gradients(d_w1)
        d_w2 = self.clip_gradients(d_w2)
        
        # Actualizacion pesos
        self.w1 += self.learning_rate * d_w1
        self.w2 += self.learning_rate * d_w2
        
        return np.mean(output_error**2)

x = np.linspace(-2, 2, 100).reshape(-1, 1)  
y = x**2 

# Normalizar
x_normalized = normalize_data(x)
y_normalized = normalize_data(y)

# Entrenamiento
nn = NeuralNetwork(input_size=1, hidden_size=10, output_size=1, learning_rate=0.001)
errors = []
iterations = []

# Entrenamiento Loop
for i in range(50000):
    # Forw
    output = nn.forward(x_normalized)
    
    # Bacj
    error = nn.backward(x_normalized, y_normalized, output)
    
    if i % 1000 == 0:
        print(f"Iteration {i}, Error MSE: {error}")
        iterations.append(i)
        errors.append(error)
        
    if error < 0.0001:  
        break

# Prediccion
final_predictions = nn.forward(x_normalized)
denormalized_predictions = denormalize_data(final_predictions, y)

# Dataframe
results = pd.DataFrame({
    'Input': x.flatten(),
    'Expected': y.flatten(),
    'Predicted': denormalized_predictions.flatten()
})

print("\nFinal Results:")
print(results)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(iterations, errors)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Training Error Over Time')

# Prediccion vs actual
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Actual', alpha=0.5)
plt.scatter(x, denormalized_predictions, label='Predicted', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predictions vs Actual')
plt.legend()
plt.tight_layout()
plt.show()

np.save('w1_stable.npy', nn.w1)
np.save('w2_stable.npy', nn.w2)
