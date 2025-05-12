
import joblib
import numpy as np
import pandas as pd
import os

# Ruta relativa a la carpeta de modelos
ruta_modelos = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos")

def predecir_falla(valvula_expansion, compresor, condensador, ruta_modelo=None):
    # Si no se proporciona una ruta específica, usar la ruta predeterminada
    if ruta_modelo is None:
        ruta_modelo = os.path.join(ruta_modelos, 'random_forest_model.pkl')
    
    # Cargar el modelo
    modelo = joblib.load(ruta_modelo)
    
    # Mapeo de índices a etiquetas
    mapeo_inverso = {0: 'FALLA ALTA', 1: 'FALLA BAJA', 2: 'SANO'}
    
    # Preparar los datos de entrada
    caracteristicas = np.array([[valvula_expansion, compresor, condensador]])
    
    # Realizar la predicción
    prediccion = modelo.predict(caracteristicas)
    probabilidades = modelo.predict_proba(caracteristicas)
    
    # Obtener la etiqueta de la clase predicha
    clase_predicha = mapeo_inverso[prediccion[0]]
    
    # Crear un DataFrame con las probabilidades
    df_prob = pd.DataFrame({
        'Clase': [mapeo_inverso[i] for i in range(len(probabilidades[0]))],
        'Probabilidad': probabilidades[0]
    })
    
    return {
        'prediccion': clase_predicha,
        'probabilidades': df_prob.sort_values('Probabilidad', ascending=False)
    }

# Ejemplo de uso
if __name__ == "__main__":
    # Reemplazar estos valores con los valores de gradiente reales que deseas clasificar
    valvula_expansion = 0.5
    compresor = 0.8
    condensador = 0.3
    
    resultado = predecir_falla(valvula_expansion, compresor, condensador)
    
    print(f"Predicción: {resultado['prediccion']}")
    print("\nProbabilidades:")
    print(resultado['probabilidades'])
