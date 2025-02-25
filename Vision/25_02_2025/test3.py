import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = "Img3.png"
image = cv2.imread(image_path)

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización
_, binary = cv2.threshold(gray, 200, 240, cv2.THRESH_BINARY_INV)

# Aplicar erosión para mejorar la detección
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)

# Encontrar contornos
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos sobre la imagen original
grid_detected = image.copy()
cv2.drawContours(grid_detected, contours, -1, (0, 255, 0), 1)

# Mostrar resultados
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagen Original")
axes[0].axis("off")

axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Escala de Grises")
axes[1].axis("off")

axes[2].imshow(binary, cmap="gray")
axes[2].set_title("Umbralización")
axes[2].axis("off")

axes[3].imshow(cv2.cvtColor(grid_detected, cv2.COLOR_BGR2RGB))
axes[3].set_title(f"Cuadros Detectados: {len(contours)}")
axes[3].axis("off")

plt.show()
