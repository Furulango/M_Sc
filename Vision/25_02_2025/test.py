import cv2
import numpy as np

def contorno_objeto(img):
    # Convertir imagen a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro de mediana
    img_gray = cv2.medianBlur(img_gray, 5)
    # Detectar bordes
    img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Encontrar contornos
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Encontrar el contorno con mayor area
    max_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest_contour = approx
                max_area = area
    # Dibujar contorno
    cv2.drawContours(img, [biggest_contour], 0, (0, 255, 0), 3)
    return img 

def dibujar_cuadricula_de_lineas_blancas(img):
    # Convertir imagen a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro de mediana
    img_gray = cv2.medianBlur(img_gray, 5)
    # Detectar bordes
    img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Encontrar contornos
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Encontrar el contorno con mayor area
    max_area = 0
    biggest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest_contour = approx
                max_area = area
    # Dibujar contorno
    cv2.drawContours(img, [biggest_contour], 0, (0, 255, 0), 3)
    # Encontrar puntos de la cuadricula
    pts1 = np.float32(biggest_contour)
    pts2 = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])
    # Encontrar matriz de transformacion
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Aplicar transformacion
    img_output = cv2.warpPerspective(img, matrix, (500, 500))
    return img_output

img = cv2.imread('img2.png')
img_cuadricula = contorno_objeto(img)
cv2.imshow('Imagen con cuadricula', img_cuadricula)
cv2.waitKey(0)







