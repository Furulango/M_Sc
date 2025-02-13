import cv2
import numpy as np  
import funciones as fn

#Contar contornos
im = cv2.imread('image.jpg')

binn = fn.conteo_cv(im)

cv2.imshow("original",im)
cv2.imshow("Otra",binn)
cv2.waitKey(0)
