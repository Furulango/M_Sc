
import cv2
import numpy as np

def conteo_cv(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bimm = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(bimm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contornos, -1, (0, 255, 0), 3)
    cv2.putText(img, 'Conteo: '+str(len(contornos)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def umbral(img, umbral):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bimm = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    return bimm

def del_background(img):
    bimm = cv2.createBackgroundSubtractorMOG2()
    mask = bimm.apply(img)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convertir a tres canales

def formas(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u, bimm = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)
    contornos, x = cv2.findContours(bimm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.04*perimetro, True)
        xd = len(aprox)
        x,y,w,h = cv2.boundingRect(aprox)
        cv2.drawContours(img, [aprox], 0, (0, 255, 0), 3)
        if xd == 3:
            cv2.putText(img, 'Triangulo', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 4:
            cv2.putText(img, 'Cuadrado', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 5:
            cv2.putText(img, 'Pentagono', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 6:
            cv2.putText(img, 'Hexagono', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, 'Otro', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def formas_color(img):
    amai = np.array([20, 90, 90], np.uint8) #Hue, Saturation, Value
    amas = np.array([70, 255, 255], np.uint8) 
    verdei = np.array([50, 100, 90], np.uint8) #Hue, Saturation, Value
    verdes = np.array([100, 255, 255], np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ama = cv2.inRange(hsv, amai, amas)
    ver = cv2.inRange(hsv, verdei, verdes)
    amarillo = cv2.bitwise_and(img, img, mask=ama)
    verde = cv2.bitwise_and(img, img, mask=ver)
    
    return img

def identificarColor_y_Forma(img):
    amai = np.array([20, 90, 90], np.uint8) #Hue, Saturation, Value
    amas = np.array([70, 255, 255], np.uint8) 
    verdei = np.array([50, 100, 90], np.uint8) #Hue, Saturation, Value
    verdes = np.array([100, 255, 255], np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ama = cv2.inRange(hsv, amai, amas)
    ver = cv2.inRange(hsv, verdei, verdes)
    amarillo = cv2.bitwise_and(img, img, mask=ama)
    verde = cv2.bitwise_and(img, img, mask=ver)
    
    # Mostrar el color en texto en la ubicación de la máscara
    contours_ama, _ = cv2.findContours(ama, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_ama:
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, 'Amarillo', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    contours_ver, _ = cv2.findContours(ver, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_ver:
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, 'Verde', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Formas
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u, bimm = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY)
    contornos, x = cv2.findContours(bimm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.04*perimetro, True)
        xd = len(aprox)
        x, y, w, h = cv2.boundingRect(aprox)
        cv2.drawContours(img, [aprox], 0, (0, 255, 0), 3)
        if xd == 3:
            cv2.putText(img, 'Triangulo', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 4:
            cv2.putText(img, 'Cuadrado', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 5:
            cv2.putText(img, 'Pentagono', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif xd == 6:
            cv2.putText(img, 'Hexagono', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, 'Otro', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def identificar_color(img):
    amai = np.array([20, 90, 90], np.uint8) #Hue, Saturation, Value
    amas = np.array([70, 255, 255], np.uint8) 
    verdei = np.array([50, 100, 90], np.uint8) #Hue, Saturation, Value
    verdes = np.array([100, 255, 255], np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ama = cv2.inRange(hsv, amai, amas)
    ver = cv2.inRange(hsv, verdei, verdes)
    amarillo = cv2.bitwise_and(img, img, mask=ama)
    verde = cv2.bitwise_and(img, img, mask=ver)
    
    # Mostrar el color en texto en la ubicación debajo de la mascara
    contours_ama, _ = cv2.findContours(ama, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_ama:
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, 'Amarillo', (x*2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    contours_ver, _ = cv2.findContours(ver, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_ver:
        x, y, w, h = cv2.boundingRect(c)
        cv2.putText(img, 'Verde', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img