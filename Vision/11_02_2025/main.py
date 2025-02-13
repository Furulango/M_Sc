import cv2
import numpy as np
import funciones as fn

img = cv2.imread("image5.jpg")
fig = fn.identificar_color(img)
fig2 = fn.formas(fig)
cv2.imshow("Formas", fig2)
cv2.waitKey(0)
