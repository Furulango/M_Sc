import cv2 as cv
import numpy as np
import os



def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def augment_img_Group1(carpeta, imagenes):
    for imagen in imagenes:
        img = cv.imread(carpeta + imagen)
        print("Imagen num: " + imagen)    
        
        # Técnicas originales
        t1 = cv.flip(img, 1)  # Volteo horizontal
        t2 = cv.add(img, np.ones(img.shape, dtype=np.uint8)*100)  # Aumentar brillo
        t3 = cv.subtract(img, np.ones(img.shape, dtype=np.uint8)*100)  # Disminuir brillo
        t4 = cv.resize(img[30:400, 20:400], (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)  # Recorte y redimensión
        
        # Técnicas adicionales
        # 1. Rotación de 90 grados
        t5 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        
        # 2. Rotación de -90 grados (270 grados)
        t6 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # 3. Espejo vertical (flip vertical)
        t7 = cv.flip(img, 0)
        
        # 4. Desenfoque gaussiano
        t8 = cv.GaussianBlur(img, (15, 15), 0)
        
        # 5. Ajuste de contraste
        alpha = 1.5  # Factor de contraste
        beta = 0  # Factor de brillo
        t9 = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        t10 = img.copy()
        salt_mask = np.random.random(img.shape[:2]) < 0.02
        t10[salt_mask] = [255, 255, 255]
        pepper_mask = np.random.random(img.shape[:2]) < 0.02
        t10[pepper_mask] = [0, 0, 0]
        
        # Guardar las imágenes originales
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/flipped_" + imagen, t1)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/brightened_" + imagen, t2)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/darkened_" + imagen, t3)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/cropped_" + imagen, t4)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/rotated90_" + imagen, t5)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/rotated270_" + imagen, t6)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/flipped_vertical_" + imagen, t7)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/blurred_" + imagen, t8)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/contrast_" + imagen, t9)
        cv.imwrite("C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/Group2/cat/salt_pepper_" + imagen, t10)

def main():
    carpeta = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales/cat"
    imagenes = os.listdir(carpeta)
    augment_img_Group1(carpeta,imagenes)

main()

