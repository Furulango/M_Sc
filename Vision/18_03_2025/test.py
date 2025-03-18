import cv2 as cv
import os

def resize_images(input_folder, output_folder, size=(100, 100)):
    # Asegurar que la carpeta de salida existe
    os.makedirs(output_folder, exist_ok=True)

    # Obtener lista de archivos en la carpeta
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv.imread(img_path)

        if img is None:
            print(f"Error cargando: {img_path}")
            continue

        resized_img = cv.resize(img, size)
        
        output_path = os.path.join(output_folder, filename)
        cv.imwrite(output_path, resized_img)

    print("Redimensionamiento completado.")

# Rutas de entrada y salida
input_folder = "C:/Users/gcmed/Downloads/Imagen/cats_set/renamed_images/"
output_folder = "C:/Users/gcmed/OneDrive/Documentos/GitHub/M_Sc/Vision/18_03_2025/Img/Originales/"

resize_images(input_folder, output_folder)
