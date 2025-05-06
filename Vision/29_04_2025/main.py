import cv2 as cv
import numpy as np
import os 

tmax=37.7
tmin=25.5

img = cv.imread('image.jpg')
h,w,c = img.shape
print("Alto:" + str(h) + " Ancho:" + str(w) + " Canales:" + str(c))
tr = np.zeros((h,w),np.float32)
for i in range(0,h):
    for j in range(0,w):
        b,g,r = img[i,j]
        #print("B:" + str(b) + " G:" + str(g) + " R:" + str(r))
        tr[i,j] = tmin + ((img[i,j,0] / 255.0) * (tmax - tmin))
a = cv.selectROI(img)
pi = img[int(a[1]):int(a[1]+a[3]), int(a[0]):int(a[0]+a[2])]
prom = 0
con = 0
for i in range(a[1], a[1]+a[3]):
    for j in range(a[0], a[0]+a[2]):
        prom += tr[i,j]
        con += 1
prom /= con
print("Temperatura promedio: " + str(prom))
cv.imshow("Imagen", img)
cv.imshow("Temperatura", tr)
cv.waitKey(0)
cv.destroyAllWindows()
