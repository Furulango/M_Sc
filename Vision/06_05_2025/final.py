import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Directorios principales
directorio_base = r"D:\GitHub\M_Sc\Vision\Material\IMAGENES TERMICAS"
carpetas = ["FALLA ALTA", "FALLA BAJA", "SANO"]

def gris_a_temperatura(valor_gris):
    return 15 + (valor_gris / 255) * (40 - 15)

def seleccionar_rois(imagen, carpeta_nombre):
    img_display = imagen.copy()
    dimensiones_roi = [(51, 52), (109, 150), (76, 211)]
    rois = []
    roi_names = ["Valvula_de_expansion", "Compresor", "Condensador"]
    
    print(f"\nSelección de ROIs para carpeta: {carpeta_nombre}")
    i = 0
    while i < len(roi_names):
        name = roi_names[i]
        # Resetear la imagen de visualización para cada nuevo intento
        img_display = imagen.copy()
        
        # Dibujar los ROIs ya seleccionados
        for j in range(i):
            x, y, w, h = rois[j]
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_display, roi_names[j], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(f"Selecciona el centro de la región {name} (haz clic y presiona ENTER)")
        print("Presiona 'r' para reintentar, 'c' para cancelar este ROI, o 'p' para usar ROI predefinido")
        cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", img_display)
        
        centro_seleccionado = []
        def click_event(event, x, y, flags, param):
            nonlocal centro_seleccionado
            if event == cv2.EVENT_LBUTTONDOWN:
                centro_seleccionado = [x, y]
                temp_img = img_display.copy()
                cv2.circle(temp_img, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", temp_img)
        
        cv2.setMouseCallback(f"Seleccionar Centro de ROI - {carpeta_nombre}", click_event)
        
        cancelar = False
        usar_predefinido = False
        while not centro_seleccionado and not cancelar and not usar_predefinido:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == ord('r'):  # Reintentar
                centro_seleccionado = []
                cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", img_display)
            elif key == ord('c'):  # Cancelar y volver al ROI anterior
                cancelar = True
                if i > 0:
                    i -= 1
                    rois.pop()  # Eliminar el último ROI
            elif key == ord('p'):  # Usar ROI predefinido
                usar_predefinido = True
        
        if usar_predefinido:
            predefined_rois = [(147, 27, 51, 52), (182, 325, 109, 150), (411, 269, 76, 211)]
            roi = predefined_rois[i]
            rois.append(roi)
            x, y, w, h = roi
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", img_display)
            print(f"Usando coordenadas predefinidas para {name}")
            i += 1
        elif centro_seleccionado and not cancelar:
            w, h = dimensiones_roi[i]
            x = max(0, centro_seleccionado[0] - w // 2)
            y = max(0, centro_seleccionado[1] - h // 2)
            
            if x + w > imagen.shape[1]:
                x = imagen.shape[1] - w
            if y + h > imagen.shape[0]:
                y = imagen.shape[0] - h
                
            roi = (x, y, w, h)
            rois.append(roi)
            
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", img_display)
            i += 1
        elif cancelar:
            continue  # Volver al ROI anterior
        else:
            # Si no se seleccionó nada y no se canceló, usar predefinido
            predefined_rois = [(147, 27, 51, 52), (182, 325, 109, 150), (411, 269, 76, 211)]
            roi = predefined_rois[i]
            rois.append(roi)
            x, y, w, h = roi
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(f"Seleccionar Centro de ROI - {carpeta_nombre}", img_display)
            print(f"Usando coordenadas predefinidas para {name}")
            i += 1
    
    # Mostrar todos los ROIs seleccionados
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return rois

def listar_archivos_ordenados(ruta_carpeta):
    """Lista los archivos de la carpeta en orden numérico."""
    if not os.path.exists(ruta_carpeta):
        print(f"La ruta {ruta_carpeta} no existe.")
        return []
    
    # Intentamos diferentes patrones para ser más flexibles
    archivos = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Intentamos diferentes patrones
            if archivo.isdigit() and archivo.endswith('.jpg'):
                # Formato simple como "1.jpg"
                numero = int(archivo.split('.')[0])
                archivos.append((numero, archivo))
            elif archivo.startswith('FLIR') and archivo[4:].isdigit():
                # Formato FLIR seguido de números
                numero = int(archivo[4:].split('.')[0])
                archivos.append((numero, archivo))
            elif archivo[0].isdigit():
                # Cualquier otro formato que comience con números
                match = re.search(r'^(\d+)', archivo)
                if match:
                    numero = int(match.group(1))
                    archivos.append((numero, archivo))
    
    archivos.sort(key=lambda x: x[0])  # Ordena por el número
    
    print(f"\nArchivos encontrados en {os.path.basename(ruta_carpeta)} ({len(archivos)}):")
    for _, archivo in archivos:
        print(f"{archivo}")
    
    return [archivo for _, archivo in archivos]

def calculo_gradiente_temperatura(resultados):
    gradientes = {}
    for roi_name, temps in resultados.items():
        if temps:  # Asegurarse de que hay temperaturas
            gradientes[roi_name] = [temps[i] - temps[0] for i in range(len(temps))]
        else:
            gradientes[roi_name] = []
    return gradientes

def procesar_imagenes(directorio_carpeta, rois, roi_names):
    # Crear directorios para cada ROI
    for i, name in enumerate(roi_names):
        roi_dir = os.path.join(directorio_carpeta, name)
        if not os.path.exists(roi_dir):
            os.makedirs(roi_dir)
    
    # Ordenar las imágenes por número
    archivos_ordenados = listar_archivos_ordenados(directorio_carpeta)
    
    if not archivos_ordenados:
        print(f"No se encontraron imágenes en {directorio_carpeta}")
        return {}, {}
    
    temperaturas_absolutas = {name: [] for name in roi_names}
    
    for i, nombre_archivo in enumerate(archivos_ordenados):
        nombre_imagen = os.path.join(directorio_carpeta, nombre_archivo)
        imagen = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)
        
        if imagen is not None:
            for j, roi in enumerate(rois):
                x, y, w, h = roi
                roi_img = imagen[y:y+h, x:x+w]
                roi_path = os.path.join(directorio_carpeta, roi_names[j], f"imagen_{i+1}.jpg")
                cv2.imwrite(roi_path, roi_img)
                valor_gris_promedio = np.mean(roi_img)
                temp_promedio = gris_a_temperatura(valor_gris_promedio)
                temperaturas_absolutas[roi_names[j]].append(temp_promedio)
                print(f"Carpeta: {os.path.basename(directorio_carpeta)}, Imagen {i+1}, {roi_names[j]}: Temperatura promedio = {temp_promedio:.2f}°C")
        else:
            print(f"No se pudo leer la imagen {nombre_imagen}")
    
    gradientes = calculo_gradiente_temperatura(temperaturas_absolutas.copy())
    
    return temperaturas_absolutas, gradientes

def graficar_resultados(gradientes, nombre_carpeta, directorio_salida):
    plt.figure(figsize=(12, 6))
    for roi_name, temps in gradientes.items():
        plt.plot(range(1, len(temps)+1), temps, label=roi_name)
    plt.title(f"Gradiente de temperatura por ROI - {nombre_carpeta}")
    plt.xlabel("Número de imagen")
    plt.ylabel("Gradiente de temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directorio_salida, f"resultados_gradiente_{nombre_carpeta.replace(' ', '_')}.png"))
    plt.close()

def guardar_resultados_csv(temperaturas, gradientes, nombre_carpeta, directorio_salida, roi_names):
    # Guardar temperaturas absolutas
    with open(os.path.join(directorio_salida, f"resultados_temperatura_{nombre_carpeta.replace(' ', '_')}.csv"), "w") as f:
        f.write("Imagen," + ",".join(roi_names) + "\n")
        max_len = max([len(temps) for temps in temperaturas.values()]) if temperaturas else 0
        for i in range(max_len):
            f.write(f"{i+1}")
            for roi_name in roi_names:
                if i < len(temperaturas[roi_name]):
                    f.write(f",{temperaturas[roi_name][i]:.2f}")
                else:
                    f.write(",")
            f.write("\n")
    
    # Guardar gradientes
    with open(os.path.join(directorio_salida, f"resultados_gradiente_{nombre_carpeta.replace(' ', '_')}.csv"), "w") as f:
        f.write("Imagen," + ",".join(roi_names) + "\n")
        max_len = max([len(temps) for temps in gradientes.values()]) if gradientes else 0
        for i in range(max_len):
            f.write(f"{i+1}")
            for roi_name in roi_names:
                if i < len(gradientes[roi_name]):
                    f.write(f",{gradientes[roi_name][i]:.2f}")
                else:
                    f.write(",")
            f.write("\n")

def guardar_gradientes_unificados(resultados_por_carpeta, directorio_base, roi_names):
    """
    Guarda todos los datos de gradientes de todas las carpetas en un único archivo CSV.
    
    Args:
        resultados_por_carpeta: Diccionario con los resultados por carpeta
        directorio_base: Directorio base donde guardar el archivo
        roi_names: Nombres de las regiones de interés
    """
    # Crear el archivo para todos los gradientes
    ruta_csv_gradientes = os.path.join(directorio_base, "gradientes_.csv")
    with open(ruta_csv_gradientes, "w") as f:
        # Escribir encabezado
        encabezado = "Carpeta,Imagen"
        for roi_name in roi_names:
            encabezado += f",{roi_name}"
        f.write(encabezado + "\n")
        
        # Escribir datos para cada carpeta
        for carpeta, (_, gradientes) in resultados_por_carpeta.items():
            max_len = max([len(temps) for temps in gradientes.values()]) if gradientes else 0
            for i in range(max_len):
                linea = f"{carpeta},{i+1}"
                for roi_name in roi_names:
                    if i < len(gradientes[roi_name]):
                        linea += f",{gradientes[roi_name][i]:.2f}"
                    else:
                        linea += ","
                f.write(linea + "\n")
    
    print(f"\nDatos de gradientes guardados en: {ruta_csv_gradientes}")
    
    return ruta_csv_gradientes

def graficar_comparativo(resultados_por_carpeta, directorio_base, roi_names):
    # Crear un único gráfico con 3 subplots (uno por ROI)
    fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    colores = {'FALLA ALTA': 'red', 'FALLA BAJA': 'orange', 'SANO': 'green'}
    
    for roi_idx, roi_name in enumerate(roi_names):
        ax = axs[roi_idx]
        
        for carpeta, (_, gradientes) in resultados_por_carpeta.items():
            if roi_name in gradientes and gradientes[roi_name]:
                color = colores.get(carpeta, None)
                ax.plot(range(1, len(gradientes[roi_name])+1), gradientes[roi_name], label=carpeta, color=color)
        
        ax.set_title(f"Comparativo de gradiente de temperatura - {roi_name}")
        ax.set_ylabel("Gradiente de temperatura (°C)")
        ax.legend()
        ax.grid(True)
    
    # Establecer la etiqueta del eje x solo para el gráfico inferior
    axs[2].set_xlabel("Número de imagen")
    
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_base, "comparativo_gradientes.png"))
    plt.show()  # Mostrar el gráfico en pantalla

def main():
    roi_names = ["Valvula_de_expansion", "Compresor", "Condensador"]
    
    usar_roi_predefinidos = input("¿Usar ROIs predefinidos para todas las carpetas? (s/n): ").lower() == 's'
    resultados_por_carpeta = {}
    
    for carpeta in carpetas:
        directorio_carpeta = os.path.join(directorio_base, carpeta)
        print(f"\nProcesando carpeta: {carpeta}")
        
        if not os.path.exists(directorio_carpeta):
            print(f"El directorio {directorio_carpeta} no existe.")
            continue
        
        # Encontrar la última imagen para seleccionar ROIs
        archivos_ordenados = listar_archivos_ordenados(directorio_carpeta)
        
        if not archivos_ordenados:
            print(f"No se encontraron imágenes en {directorio_carpeta}")
            continue
        
        ultima_imagen = os.path.join(directorio_carpeta, archivos_ordenados[-1])
        imagen = cv2.imread(ultima_imagen, cv2.IMREAD_GRAYSCALE)
        
        if imagen is not None:
            if usar_roi_predefinidos:
                rois = [(147, 27, 51, 52), (182, 325, 109, 150), (411, 269, 76, 211)]
                print(f"Usando regiones predefinidas para {carpeta}:")
                print(rois)
            else:
                rois = seleccionar_rois(imagen, carpeta)
                print(f"Regiones seleccionadas para {carpeta}:")
                print(rois)
            
            temperaturas_absolutas, gradientes = procesar_imagenes(directorio_carpeta, rois, roi_names)
            resultados_por_carpeta[carpeta] = (temperaturas_absolutas, gradientes)
            
            # Graficar y guardar resultados individuales
            graficar_resultados(gradientes, carpeta, directorio_carpeta)
            guardar_resultados_csv(temperaturas_absolutas, gradientes, carpeta, directorio_carpeta, roi_names)
            
            print(f"Procesamiento de {carpeta} completo. Resultados guardados.")
        else:
            print(f"No se pudo leer la imagen {ultima_imagen}")
    
    # Guardar todos los datos de gradientes en un único archivo CSV
    if resultados_por_carpeta:
        print("\nGuardando todos los datos de gradientes en un archivo CSV unificado...")
        ruta_archivo_gradientes = guardar_gradientes_unificados(resultados_por_carpeta, directorio_base, roi_names)
        print(f"Archivo CSV de gradientes guardado en: {ruta_archivo_gradientes}")

    
    # Crear gráficos comparativos entre carpetas
    if len(resultados_por_carpeta) > 1:
        print("\nGenerando gráficos comparativos...")
        graficar_comparativo(resultados_por_carpeta, directorio_base, roi_names)
        print("Gráficos comparativos generados.")
    
    print("\nProcesamiento de todas las carpetas completado.")

if __name__ == "__main__":
    main()