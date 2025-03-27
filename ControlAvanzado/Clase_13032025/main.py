import os
import funciones as f
file_path = input("Ingrese la ruta del archivo de datos: ")
if os.path.exists(file_path):
    f.main(file_path)
else:
    print(f"El archivo {file_path} no existe.")