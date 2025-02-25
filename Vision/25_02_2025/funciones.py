import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(ruta_imagen):
    return cv2.imread(ruta_imagen)

def convertir_a_grises(imagen):
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def aplicar_umbral(imagen_gris, valor_umbral=190, valor_max=240):
    _, binaria = cv2.threshold(imagen_gris, valor_umbral, valor_max, cv2.THRESH_BINARY_INV)
    return binaria

def aplicar_erosion(imagen_binaria, tamano_kernel=(3, 3), iteraciones=1):
    kernel = np.ones(tamano_kernel, np.uint8)
    return cv2.erode(imagen_binaria, kernel, iterations=iteraciones)

def encontrar_contornos(imagen_erosionada):
    contornos, _ = cv2.findContours(imagen_erosionada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos

def dibujar_contornos(imagen, contornos):
    copia_imagen = imagen.copy()
    cv2.drawContours(copia_imagen, contornos, -1, (0, 255, 0), 1)
    return copia_imagen

def mostrar_resultados(imagenes, titulos):
    fig, axes = plt.subplots(1, len(imagenes), figsize=(16, 4))
    for ax, img, titulo in zip(axes, imagenes, titulos):
        cmap = "gray" if len(img.shape) == 2 else None
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if cmap is None else img, cmap=cmap)
        ax.set_title(titulo)
        ax.axis("off")
    plt.show()

def main(ruta_imagen):
    imagen = cargar_imagen(ruta_imagen)
    gris = convertir_a_grises(imagen)
    binaria = aplicar_umbral(gris)
    erosionada = aplicar_erosion(binaria)
    contornos = encontrar_contornos(erosionada)
    cuadricula_detectada = dibujar_contornos(imagen, contornos)
    
    mostrar_resultados(
        [imagen, gris, binaria, cuadricula_detectada],
        ["Imagen Original", "Escala de Grises", "Umbralización", f"Cuadros Detectados: {len(contornos)}"]
    )

