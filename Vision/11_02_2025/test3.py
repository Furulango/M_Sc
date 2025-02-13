import cv2
import numpy as np

def remove_background(img):
    """
    Remueve el fondo de una imagen usando OpenCV.
    
    Args:
        img: Imagen cargada con cv2.imread()
    
    Returns:
        imagen_sin_fondo: Imagen con fondo transparente (canal alpha)
    """
    # Convertir a RGBA (agregar canal alpha)
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        img_rgba = cv2.merge((b, g, r, alpha))
    else:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque Gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Obtener umbral usando el método de Otsu
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear máscara
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    
    # Dibujar contornos en la máscara
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Filtrar contornos pequeños
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
    
    # Suavizar máscara
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Aplicar máscara al canal alpha
    img_rgba[:, :, 3] = mask
    
    return img_rgba

# Ejemplo de uso:
img = cv2.imread("image4.jpg")
resultado = remove_background(img)
cv2.imwrite("resultado.png", resultado)  # Guardar como PNG para preservar transparencia