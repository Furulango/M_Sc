import cv2
import numpy as np
import funciones as fn

im = cv2.imread(r'Images\test2.jpg')  
h, w, ch = im.shape
grises = fn.grises(im,h,w)

summ = fn.suma(grises,h,w,100)
multi = fn.multiplicacion(grises,h,w,4)
cv2.imshow('Original', im)
cv2.imshow('Grises', grises)
cv2.imshow('Suma', summ)
cv2.imshow('Multiplicacion', multi)
cv2.waitKey(0)
cv2.destroyAllWindows()
