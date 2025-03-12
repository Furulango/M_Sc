import cv2 as cv
import numpy as np
import os

carpeta = "C:/Users/gcmed/Downloads/Imagen/cats_set/renamed_images/"
imagenes = os.listdir(carpeta)

def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

for imagen in imagenes:
    img = cv.imread(carpeta + imagen)
    
    h, w, c = img.shape
    print("Imagen num: " + imagen)    
    t1 = cv.flip(img, 1)
    t2 = cv.add(img, np.ones(img.shape, dtype=np.uint8)*100)
    t3 = cv.subtract(img, np.ones(img.shape, dtype=np.uint8)*100)
    t4 = cv.resize(img[30:400, 20:400], (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

    resize_width = 200
    
    img_resized = resize_image(img, resize_width)
    t1_resized = resize_image(t1, resize_width)
    t2_resized = resize_image(t2, resize_width)
    t3_resized = resize_image(t3, resize_width)
    t4_resized = resize_image(t4,resize_width)
    
    combined = np.hstack((img_resized, t1_resized, t2_resized, t3_resized, t4_resized))
    cv.imwrite("Images_results/combined_" + imagen, combined)    
    cv.waitKey(0)
    cv.destroyAllWindows()
