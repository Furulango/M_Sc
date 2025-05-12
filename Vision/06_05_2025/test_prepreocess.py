import os
import re
import shutil

def listar_archivos_ordenados(ruta_carpeta):
    """Lista los archivos de la carpeta en orden numérico del FLIR#### al FLIR####."""
    if not os.path.exists(ruta_carpeta):
        print(f"La ruta {ruta_carpeta} no existe.")
        return []
    
    # Patrón para archivos FLIR seguidos de números
    patron = re.compile(r'^(FLIR)(\d+)(\..+)$')
    archivos = []
    
    for archivo in os.listdir(ruta_carpeta):
        match = patron.match(archivo)
        if match:
            prefijo = match.group(1)  # FLIR
            numero = int(match.group(2))  # El número después de FLIR
            extension = match.group(3)  # La extensión con el punto
            archivos.append((numero, archivo, prefijo, extension))
    
    archivos.sort()  # Ordena por el número
    
    print(f"\nArchivos encontrados ({len(archivos)}):")
    print("-" * 50)
    for numero, archivo, prefijo, extension in archivos:
        print(f"{archivo}")
    print("-" * 50)
    
    return archivos

def eliminar_pares_y_renombrar(ruta_carpeta):
    if not os.path.exists(ruta_carpeta):
        print(f"La ruta {ruta_carpeta} no existe.")
        return
    
    # Llamamos a la función para listar archivos antes de cualquier operación
    archivos = listar_archivos_ordenados(ruta_carpeta)
    
    if not archivos:
        print("No se encontraron archivos para procesar.")
        return
    
    archivos_eliminados = 0
    archivos_impares = []
    
    # Consideramos el índice relativo en lugar del número en el nombre
    for i, (numero, archivo, prefijo, extension) in enumerate(archivos, 1):
        if i % 2 == 0:  # Eliminamos los de índice par (segundo, cuarto, etc.)
            ruta_completa = os.path.join(ruta_carpeta, archivo)
            try:
                os.remove(ruta_completa)
                print(f"Eliminado: {archivo} (posición par)")
                archivos_eliminados += 1
            except Exception as e:
                print(f"Error al eliminar {archivo}: {e}")
        else:
            archivos_impares.append((numero, archivo, prefijo, extension))
    
    print(f"\nSe eliminaron {archivos_eliminados} archivos en posiciones pares.")
    
    carpeta_temp = os.path.join(ruta_carpeta, "temp_renombrado")
    if not os.path.exists(carpeta_temp):
        os.makedirs(carpeta_temp)
    
    for numero, archivo, prefijo, extension in archivos_impares:
        ruta_original = os.path.join(ruta_carpeta, archivo)
        ruta_temp = os.path.join(carpeta_temp, archivo)
        shutil.copy2(ruta_original, ruta_temp)
    
    for numero, archivo, prefijo, extension in archivos_impares:
        ruta_original = os.path.join(ruta_carpeta, archivo)
        os.remove(ruta_original)
    
    print("\nRenombrando archivos restantes...")
    for i, (numero, archivo, prefijo, extension) in enumerate(archivos_impares, 1):
        ruta_vieja = os.path.join(carpeta_temp, archivo)
        nuevo_nombre = f"{i}{extension}"
        ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)
        shutil.copy2(ruta_vieja, ruta_nueva)
        print(f"Renombrado: {archivo} -> {nuevo_nombre}")
    
    shutil.rmtree(carpeta_temp)
    
    print(f"\nProceso completado. Los {len(archivos_impares)} archivos impares han sido renombrados del 1 al {len(archivos_impares)}.")

ruta = r"C:\Users\gumev\Desktop\IMAGENES TERMICAS\FALLA BAJA"
eliminar_pares_y_renombrar(ruta)