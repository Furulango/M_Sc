import cv2 as cv
import numpy as np
import os

def resize_image(image, width):
    aspect_ratio = width / image.shape[1]
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def augment_img_Group1(carpeta, imagenes):
    for imagen in imagenes:
        # Asegúrate de que los archivos tengan la extensión ".png"
        if not imagen.lower().endswith('.png'):
            print(f"Archivo ignorado (no PNG): {imagen}")
            continue
        
        img_path = os.path.join(carpeta, imagen)
        
        # Cargar la imagen
        img = cv.imread(img_path)
        
        # Verificar si la imagen fue cargada correctamente
        if img is None:
            print(f"Error al cargar la imagen: {img_path}")
            continue  # Salta a la siguiente imagen si no se pudo cargar
        
        print(f"Imagen num: {imagen}")
        
        # Aplicar transformaciones
        t1 = cv.flip(img, 1)
        t2 = cv.add(img, np.ones(img.shape, dtype=np.uint8)*100)
        t3 = cv.subtract(img, np.ones(img.shape, dtype=np.uint8)*100)
        t4 = cv.resize(img[30:400, 20:400], (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)
        
        # Guardar las imágenes procesadas
        output_dir = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/11_03_2025/Images_augm/TEST/dog"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv.imwrite(os.path.join(output_dir, f"flipped_{imagen}"), t1)
        cv.imwrite(os.path.join(output_dir, f"brightened_{imagen}"), t2)
        cv.imwrite(os.path.join(output_dir, f"darkened_{imagen}"), t3)
        cv.imwrite(os.path.join(output_dir, f"cropped_{imagen}"), t4)

def main():
    carpeta = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales/dog"
    imagenes = os.listdir(carpeta)
    augment_img_Group1(carpeta, imagenes)

main()
