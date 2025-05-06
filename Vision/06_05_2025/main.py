import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

directorio_imagenes = r"C:\Users\gumev\Desktop\IMAGENES TERMICAS\SANO"

def gris_a_temperatura(valor_gris):
    return 15 + (valor_gris / 255) * (40 - 15)

def seleccionar_rois(imagen):
    img_display = imagen.copy()
    dimensiones_roi = [(51, 52), (109, 150), (76, 211)]
    rois = []
    roi_names = ["ROI_1", "ROI_2", "ROI_3"]
    
    for i, name in enumerate(roi_names):
        print(f"Selecciona el centro de la región {name} (haz clic y presiona ENTER)")
        cv2.imshow("Seleccionar Centro de ROI", img_display)
        
        centro_seleccionado = []
        def click_event(event, x, y, flags, param):
            nonlocal centro_seleccionado
            if event == cv2.EVENT_LBUTTONDOWN:
                centro_seleccionado = [x, y]
                cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Seleccionar Centro de ROI", img_display)
        
        cv2.setMouseCallback("Seleccionar Centro de ROI", click_event)
        
        while not centro_seleccionado:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
        
        if centro_seleccionado:
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
            cv2.imshow("Seleccionar Centro de ROI", img_display)
        else:
            predefined_rois = [(147, 27, 51, 52), (182, 325, 109, 150), (411, 269, 76, 211)]
            rois.append(predefined_rois[i])
            x, y, w, h = predefined_rois[i]
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_display, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Usando coordenadas predefinidas para {name}")
    
    cv2.destroyAllWindows()
    return rois

def procesar_imagenes(rois):
    for i in range(len(rois)):
        roi_dir = os.path.join(directorio_imagenes, f"ROI_{i+1}")
        if not os.path.exists(roi_dir):
            os.makedirs(roi_dir)
    temperaturas_absolutas = {f"ROI_{i+1}": [] for i in range(len(rois))}
    for i in range(1, 61):
        nombre_imagen = os.path.join(directorio_imagenes, f"{i}.jpg")
        if os.path.exists(nombre_imagen):
            imagen = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)
            if imagen is not None:
                for j, roi in enumerate(rois):
                    x, y, w, h = roi
                    roi_img = imagen[y:y+h, x:x+w]
                    roi_path = os.path.join(directorio_imagenes, f"ROI_{j+1}", f"imagen_{i}.jpg")
                    cv2.imwrite(roi_path, roi_img)
                    valor_gris_promedio = np.mean(roi_img)
                    temp_promedio = gris_a_temperatura(valor_gris_promedio)
                    temperaturas_absolutas[f"ROI_{j+1}"].append(temp_promedio)
                    print(f"Imagen {i}, ROI_{j+1}: Temperatura promedio = {temp_promedio:.2f}°C")
            else:
                print(f"No se pudo leer la imagen {nombre_imagen}")
        else:
            print(f"La imagen {nombre_imagen} no existe")
    
    gradientes = calculo_gradiente_temperatura(temperaturas_absolutas.copy())
    
    return temperaturas_absolutas, gradientes

def graficar_resultados(gradientes):
    plt.figure(figsize=(12, 6))
    for roi_name, temps in gradientes.items():
        plt.plot(range(1, len(temps)+1), temps, label=roi_name)
    plt.title("Gradiente de temperatura por ROI")
    plt.xlabel("Número de imagen")
    plt.ylabel("Gradiente de temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(directorio_imagenes, "resultados_gradiente.png"))
    plt.show()

def calculo_gradiente_temperatura(resultados):
    gradientes = {}
    for roi_name, temps in resultados.items():
        gradientes[roi_name] = [temps[i] - temps[0] for i in range(len(temps))]
    return gradientes

def main():
    usar_roi_predefinidos = input("¿Usar ROIs predefinidos? (s/n): ").lower() == 's'
    
    ultima_imagen = os.path.join(directorio_imagenes, "60.jpg")
    if os.path.exists(ultima_imagen):
        imagen = cv2.imread(ultima_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            if usar_roi_predefinidos:
                rois = [(147, 27, 51, 52), (182, 325, 109, 150), (411, 269, 76, 211)]
                print("Usando regiones predefinidas:")
                print(rois)
            else:
                rois = seleccionar_rois(imagen)
                print("Regiones seleccionadas:")
                print(rois)
            
            temperaturas_absolutas, gradientes = procesar_imagenes(rois)
            
            graficar_resultados(gradientes)
            
            with open(os.path.join(directorio_imagenes, "resultados_temperatura.csv"), "w") as f:
                f.write("Imagen," + ",".join([f"ROI_{i+1}" for i in range(len(rois))]) + "\n")
                for i in range(len(temperaturas_absolutas["ROI_1"])):
                    f.write(f"{i+1}")
                    for j in range(len(rois)):
                        f.write(f",{temperaturas_absolutas[f'ROI_{j+1}'][i]:.2f}")
                    f.write("\n")
                
            with open(os.path.join(directorio_imagenes, "resultados_gradiente.csv"), "w") as f:
                f.write("Imagen," + ",".join([f"ROI_{i+1}" for i in range(len(rois))]) + "\n")
                for i in range(len(gradientes["ROI_1"])):
                    f.write(f"{i+1}")
                    for j in range(len(rois)):
                        f.write(f",{gradientes[f'ROI_{j+1}'][i]:.2f}")
                    f.write("\n")
                
            print("Procesamiento completo. Resultados guardados.")
        else:
            print(f"No se pudo leer la imagen {ultima_imagen}")
    else:
        print(f"La imagen {ultima_imagen} no existe")

if __name__ == "__main__":
    main()
