import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('D:\GitHub\M_Sc\Vision\Material\IMAGENES TERMICAS\gradientes_.csv')

# Diccionario de traducción
translations = {
    'FALLA ALTA': 'HIGH FAULT',
    'FALLA BAJA': 'LOW FAULT',
    'SANO': 'HEALTHY'
}

# Reemplazar los valores en la columna 'Carpeta'
df['Carpeta'] = df['Carpeta'].replace(translations)

# Guardar el archivo modificado
df.to_csv('datos_traducidos.csv', index=False)

# Mostrar las primeras filas para verificar
print("Primeras 10 filas del archivo traducido:")
print(df.head(10))

# Mostrar un resumen de las categorías
print("\nConteo de categorías:")
print(df['Carpeta'].value_counts())