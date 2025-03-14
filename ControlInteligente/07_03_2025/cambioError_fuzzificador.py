import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def controlador_difuso(error_rad):
    """
    Controlador difuso que convierte un valor de error en radianes a un vector de membresía F.
    
    Args:
        error_rad (float): Valor del error en radianes.
        
    Returns:
        list: Vector F = [negativogrande, negativopequeño, cero, positivopequeño, positivogrande]
              con valores de membresía que suman exactamente 1.
    """
    # Asegurar que el error esté dentro de los límites
    error_rad = max(-np.pi/2, min(np.pi/2, error_rad))
    
    # Inicializar valores de membresía
    F = [0.0, 0.0, 0.0, 0.0, 0.0]  # [negativogrande, negativopequeño, cero, positivopequeño, positivogrande]
    
    # Calcular membresías
    if error_rad <= -np.pi/4:
        # Completamente neggrande
        F[0] = 1.0
    elif error_rad <= -np.pi/8:
        # Entre neggrande y negpequeño
        ratio = (error_rad - (-np.pi/4)) / (np.pi/8)  # 0 en -pi/4, 1 en -pi/8
        F[0] = 1 - ratio
        F[1] = ratio
    elif error_rad <= 0:
        # Entre negpequeño y cero
        ratio = (error_rad - (-np.pi/8)) / (np.pi/8)  # 0 en -pi/8, 1 en 0
        F[1] = 1 - ratio
        F[2] = ratio
    elif error_rad <= np.pi/8:
        # Entre cero y pospequeño
        ratio = (error_rad - 0) / (np.pi/8)  # 0 en 0, 1 en pi/8
        F[2] = 1 - ratio
        F[3] = ratio
    elif error_rad <= np.pi/4:
        # Entre pospequeño y posgrande
        ratio = (error_rad - (np.pi/8)) / (np.pi/8)  # 0 en pi/8, 1 en pi/4
        F[3] = 1 - ratio
        F[4] = ratio
    else:
        # Completamente posgrande
        F[4] = 1.0
    
    # Asegurar que la suma sea exactamente 1 (posiblers errores de punto flotante)
    total = sum(F)
    if total > 0:
        F = [x/total for x in F]
    
    return F

def generar_datos_prueba(num_puntos=20):
    """
    Genera datos de prueba para el controlador difuso.
    
    Args:
        num_puntos (int): Número de puntos a generar.
        
    Returns:
        numpy.ndarray: Array con los valores de error en radianes.
    """
    return np.linspace(-np.pi/4, np.pi/4, num_puntos)

def crear_dataframe(errores_prueba):
    """
    Crea un DataFrame con los resultados del controlador difuso.
    
    Args:
        errores_prueba (numpy.ndarray): Array con los valores de error a probar.
        
    Returns:
        pandas.DataFrame: DataFrame con los resultados.
    """
    # Definir etiquetas para las columnas
    etiquetas = ["NegativoGrande", "NegativoPequeño", "Cero", "PositivoPequeño", "PositivoGrande"]
    
    # Crear una lista para almacenar los resultados
    resultados = []
    
    for error in errores_prueba:
        vector_membresia = controlador_difuso(error)
        # Convertir error de radianes a grados para mejor legibilidad
        error_grados = np.degrees(error)
        # Añadir a la lista de resultados
        resultados.append([error, error_grados] + vector_membresia)
    
    # Crear un DataFrame con los resultados
    columnas = ["Error_rad", "Error_grados"] + etiquetas
    return pd.DataFrame(resultados, columns=columnas)

def mostrar_dataframe(df):
    """
    Muestra el DataFrame con los resultados del controlador difuso.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados.
    """
    # Configurar pandas para mostrar más decimales
    pd.set_option('display.precision', 4)
    
    # Mostrar el DataFrame
    print("\nResultados del controlador difuso:")
    print(df)

def guardar_dataframe(df, nombre_archivo="cambioError_fuzzificador.csv"):
    """
    Guarda el DataFrame en un archivo CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados.
        nombre_archivo (str): Nombre del archivo CSV.
    """
    df.to_csv(nombre_archivo, index=False)
    print(f"\nResultados guardados en '{nombre_archivo}'")

def crear_grafico_lineas(df, etiquetas):
    """
    Crea un gráfico de líneas con los valores de membresía.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados.
        etiquetas (list): Lista con las etiquetas de las columnas.
    """
    plt.figure(figsize=(10, 6))
    for i, etiqueta in enumerate(etiquetas):
        plt.plot(df["Error_grados"], df[etiqueta], linewidth=2, label=etiqueta)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Error (grados)")
    plt.ylabel("Grado de Membresía")
    plt.title("Funciones de Membresía del Controlador Difuso")
    plt.legend()
    plt.tight_layout()

def crear_mapa_calor(df, etiquetas):
    """
    Crea un mapa de calor con los valores de membresía.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados.
        etiquetas (list): Lista con las etiquetas de las columnas.
    """
    plt.figure(figsize=(12, 5))
    matriz_membresia = df[etiquetas].values
    sns.heatmap(matriz_membresia.T, cmap="Purple", 
                xticklabels=np.round(df["Error_grados"].values, 1),
                yticklabels=etiquetas,
                cbar_kws={'label': 'Grado de Membresía'})
    plt.xlabel("Error (grados)")
    plt.title("Mapa de Calor de Valores de Membresía")
    plt.tight_layout()

def visualizar_resultados(df):
    """
    Visualiza los resultados del controlador difuso.
    
    Args:
        df (pandas.DataFrame): DataFrame con los resultados.
    """
    etiquetas = ["NegativoGrande", "NegativoPequeño", "Cero", "PositivoPequeño", "PositivoGrande"]
    
    # Crear visualizaciones
    plt.figure(figsize=(15, 10))
    
    # 1. Gráfico de líneas: Valores de membresía vs Error
    plt.subplot(2, 1, 1)
    for i, etiqueta in enumerate(etiquetas):
        plt.plot(df["Error_grados"], df[etiqueta], linewidth=2, label=etiqueta)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Error (grados)")
    plt.ylabel("Grado de Membresía")
    plt.title("Funciones de Membresía ")
    plt.legend()
    
    # 2. Mapa de calor (heatmap) para visualizar la matriz de membresía
    plt.subplot(2, 1, 2)
    matriz_membresia = df[etiquetas].values
    sns.heatmap(matriz_membresia.T, cmap="Blues", 
                xticklabels=np.round(df["Error_grados"].values, 1),
                yticklabels=etiquetas,
                cbar_kws={'label': 'Grado de Membresía'})
    plt.xlabel("Error (grados)")
    plt.title("Mapa de Calor Membresía ")
    
    plt.tight_layout()
    plt.show()

def main():

    errores_prueba = generar_datos_prueba(5)
    
    df = crear_dataframe(errores_prueba)
    
    mostrar_dataframe(df)

    visualizar_resultados(df)
    
    # Guardar resultados
    guardar_dataframe(df)
    
if __name__ == "__main__":
    main()