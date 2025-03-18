import cv2 as cv
import numpy as np
import os



def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def augment_img_Group1(carpeta,imagenes):
    for imagen in imagenes:
        img = cv.imread(carpeta + imagen)
        print("Imagen num: " + imagen)    
        t1 = cv.flip(img, 1)
        t2 = cv.add(img, np.ones(img.shape, dtype=np.uint8)*100)
        t3 = cv.subtract(img, np.ones(img.shape, dtype=np.uint8)*100)
        t4 = cv.resize(img[30:400, 20:400], (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)
        
        #cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group1/original_" + imagen, img_resized)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/TEST/cat/flipped_" + imagen, t1)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/TEST/cat/brightened_" + imagen, t2)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/TEST/cat/darkened_" + imagen, t3)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/TEST/cat/cropped_" + imagen, t4)


def main():
    carpeta = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales/cat"
    imagenes = os.listdir(carpeta)
    augment_img_Group1(carpeta,imagenes)

main()

