import numpy as np

def grises(im, h, w):
    b = im[:, :, 0]
    g = im[:, :, 1]
    r = im[:, :, 2]

    gris = np.copy(b)

    for i in range(0, h):
        for j in range(0, w):
            gris[i, j] = 0.299 * r[i, j] + 0.587 * g[i, j] + 0.114 * b[i, j]

    return gris

def binaria(im, h, w):
    gris = grises(im, h, w)
    binaria = np.copy(gris)

    for i in range(0, h):
        for j in range(0, w):
            if gris[i, j] > 100:
                binaria[i, j] = 255
            else:
                binaria[i, j] = 0

    return binaria

def suma(gris,h,w,n):
    summ = np.zeros((h,w),np.uint8)
    for i in range (0,h):
        for j in range (0,w):
            summ[i,j] = gris[i,j] + n
            if summ[i,j] > 255:
                summ[i,j] = 255
            else:
                summ[i,j] = summ[i,j]
    return summ

#

def multiplicacion(gris,h,w,n):
    mult = np.zeros((h,w),np.uint8)
    for i in range (0,h):
        for j in range (0,w):
            mult[i,j] = gris[i,j] * n
            if mult[i,j] > 255:
                mult[i,j] = 255
            else:
                mult[i,j] = mult[i,j]
    return mult