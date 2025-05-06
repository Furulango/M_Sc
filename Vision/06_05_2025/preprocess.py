import os
import re
import shutil

def eliminar_pares_y_renombrar(ruta_carpeta):
    if not os.path.exists(ruta_carpeta):
        print(f"La ruta {ruta_carpeta} no existe.")
        return
    
    patron = re.compile(r'^(\d+)(\..+)$')
    archivos = []
    for archivo in os.listdir(ruta_carpeta):
        match = patron.match(archivo)
        if match:
            numero = int(match.group(1))
            extension = match.group(2)
            archivos.append((numero, archivo, extension))
    
    archivos.sort()
    archivos_eliminados = 0
    archivos_impares = []
    
    for numero, archivo, extension in archivos:
        if numero % 2 == 0:
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            try:
                os.remove(ruta_completa)
                print(f"Eliminado: {archivo} (número par)")
                archivos_eliminados += 1
            except Exception as e:
                print(f"Error al eliminar {archivo}: {e}")
        else:
            archivos_impares.append((numero, archivo, extension))
    
    print(f"\nSe eliminaron {archivos_eliminados} archivos con números pares.")
    
    carpeta_temp = os.path.join(ruta_carpeta, "temp_renombrado")
    if not os.path.exists(carpeta_temp):
        os.makedirs(carpeta_temp)
    
    for numero, archivo, extension in archivos_impares:
        ruta_original = os.path.join(ruta_carpeta, archivo)
        ruta_temp = os.path.join(carpeta_temp, archivo)
        shutil.copy2(ruta_original, ruta_temp)
    
    for numero, archivo, extension in archivos_impares:
        ruta_original = os.path.join(ruta_carpeta, archivo)
        os.remove(ruta_original)
    
    print("\nRenombrando archivos restantes...")
    for i, (numero, archivo, extension) in enumerate(archivos_impares, 1):
        ruta_vieja = os.path.join(carpeta_temp, archivo)
        nuevo_nombre = f"{i}{extension}"
        ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)
        shutil.copy2(ruta_vieja, ruta_nueva)
        print(f"Renombrado: {archivo} -> {nuevo_nombre}")
    
    shutil.rmtree(carpeta_temp)
    
    print(f"\nProceso completado. Los {len(archivos_impares)} archivos impares han sido renombrados del 1 al {len(archivos_impares)}.")

ruta = r"C:\Users\gumev\Desktop\IMAGENES TERMICAS\SANO\AA"
eliminar_pares_y_renombrar(ruta)
