import os
import cupy as cp
from PIL import Image
import random
import numpy as np

print("¿CuPy está usando la GPU?", cp.cuda.is_available())

gpu_info = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"GPU detectada: {props['name'].decode()} ")

# Memoria disponible
mem_info = gpu_info.mem_info
print(f"Memoria GPU disponible: {mem_info[0] / 1e9:.2f} GB / {mem_info[1] / 1e9:.2f} GB")

def load_images(carpeta, img_size=(100, 100)):
    imagenes = os.listdir(carpeta)
    dataset = []
    
    for img_name in imagenes:
        img_path = os.path.join(carpeta, img_name)
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(img_size)
            img_array = cp.array(img) / 255.0 
            dataset.append(img_array.flatten())
        except:
            print(f"Error cargando imagen: {img_path}")
    
    return cp.array(dataset)  

def add_noise(data, noise_factor=0.001):
    noise = noise_factor * cp.random.randn(*data.shape)  # Ruido
    return cp.clip(data + noise, 0., 1.)

def initialize_weights(input_size, hidden_size):
    W1 = cp.random.randn(hidden_size, input_size) * 0.01  # Pesos
    b1 = cp.zeros((hidden_size, 1))  # Sesgos
    W2 = cp.random.randn(input_size, hidden_size) * 0.01  # Pesos
    b2 = cp.zeros((input_size, 1))  # Sesgos
    return W1, b1, W2, b2

def sigmoid(x):
    return 1 / (1 + cp.exp(-x))  # Sigmoide

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivada sigmoide

def train_autoencoder(X, hidden_size=128, epochs=1000, lr=0.01):
    input_size = X.shape[1]
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size)
    
    for epoch in range(epochs):
        Z1 = cp.dot(W1, X.T) + b1
        A1 = sigmoid(Z1)
        Z2 = cp.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
        loss = cp.mean((A2 - X.T) ** 2)
        
        dA2 = A2 - X.T
        dZ2 = dA2 * sigmoid_derivative(A2)
        dW2 = cp.dot(dZ2, A1.T) / X.shape[0]
        db2 = cp.sum(dZ2, axis=1, keepdims=True) / X.shape[0]
        
        dA1 = cp.dot(W2.T, dZ2)
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = cp.dot(dZ1, X) / X.shape[0]
        db1 = cp.sum(dZ1, axis=1, keepdims=True) / X.shape[0]
        
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f}")
    
    return W1, b1, W2, b2

def reconstruct_images(X, W1, b1, W2, b2):
    Z1 = cp.dot(W1, X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = cp.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2.T  # Devolver en formato adecuado

def save_weights(W1, b1, W2, b2, filename="autoencoder_weights.npz"):
    cp.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Pesos guardados en {filename}")

def load_weights(filename="autoencoder_weights.npz"):
    data = cp.load(filename)
    W1 = data['W1']
    b1 = data['b1']
    W2 = data['W2']
    b2 = data['b2']
    print(f"Pesos cargados desde {filename}")
    return W1, b1, W2, b2

# Ruta de las imágenes
carpeta = "C:/Users/gcmed/Downloads/Imagen/cats_set/renamed_images/"

# Cargar imágenes
X = load_images(carpeta)

# Añadir ruido
X_noisy = add_noise(X)

# Verificar si los pesos ya están guardados
weights_file = "autoencoder_weights.npz"
if os.path.exists(weights_file):
    print("Cargando pesos existentes...")
    W1, b1, W2, b2 = load_weights(weights_file)
else:
    print("Entrenando el autoencoder...")
    W1, b1, W2, b2 = train_autoencoder(X_noisy, hidden_size=256, epochs=2000, lr=0.05)
    # Guardar los pesos después de entrenar
    save_weights(W1, b1, W2, b2)

# Reconstruir imágenes
X_reconstructed = reconstruct_images(X_noisy, W1, b1, W2, b2)

# Carpeta guardar imagenes
output_folder = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group3/"
os.makedirs(output_folder, exist_ok=True)

# Guardar las imágenes reconstruidas
for i, img_array in enumerate(X_reconstructed):
    img = (img_array.get() * 255).reshape(100, 100).astype(np.uint8)  # Convertir de CuPy a NumPy antes de guardar
    Image.fromarray(img).save(os.path.join(output_folder, f"reconstructed_{i}.png"))
