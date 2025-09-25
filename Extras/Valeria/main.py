import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Configuración de la Carpeta de Resultados ---

# Definir la ruta base y la carpeta de resultados.
output_dir = '/home/furulango/GitHub/M_Sc/Extras/Valeria/results'
# Crear la carpeta si no existe.
os.makedirs(output_dir, exist_ok=True)

# Definir la ruta completa para el archivo de texto de resultados.
results_txt_path = os.path.join(output_dir, 'analisis_completo.txt')

# Redirigir la salida estándar (print) a nuestro archivo de texto.
original_stdout = sys.stdout
with open(results_txt_path, 'w') as f:
    sys.stdout = f

    # --- 2. Carga y Preparación de Datos ---
    try:
        # Cargar la hoja específica del archivo Excel.
        df = pd.read_excel('/home/furulango/GitHub/M_Sc/Extras/Valeria/Hoja de cálculo sin título.xlsx', sheet_name='PARA ESTADÍSTICO')
    except FileNotFoundError:
        print("Error: El archivo no fue encontrado. Asegúrate de que el nombre del archivo es correcto y está en la misma carpeta.")
        sys.stdout = original_stdout # Restaurar stdout antes de salir
        exit()
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Excel: {e}")
        print("Asegúrate de que el archivo contiene una hoja llamada 'PARA ESTADÍSTICO' y las columnas están bien nombradas.")
        sys.stdout = original_stdout # Restaurar stdout antes de salir
        exit()

    # Convertir las variables de los factores a tipo 'category'.
    try:
        df['ANO'] = df['ANO'].astype('category')
        df['METODO_EXTRaccion'] = df['METODO_EXTRACCION'].astype('category')
    except KeyError as e:
        print(f"Error de columna: No se encontró la columna {e}.")
        print("Por favor, verifica que los nombres en tu archivo Excel coincidan exactamente con los esperados (ANO, METODO_EXTRACCION).")
        sys.stdout = original_stdout # Restaurar stdout antes de salir
        exit()

    print("--- Datos cargados correctamente ---")
    print("Columnas encontradas:", df.columns.tolist())
    print("\n" + "="*80 + "\n")


    # --- 3. ANOVA de Dos Vías y Prueba Post-Hoc de Tukey ---
    response_variables = [
        'FOLIN_MG_EAG',
        'DPPH_MG_TROLOX',
        'ABTS_MG_TROLOX',
        'FRAP_MG_TROLOX',
        'FLAVONOIDES_MG_RUTINA'
    ]

    for var in response_variables:
        if var not in df.columns:
            print(f"Advertencia: La columna '{var}' no se encontró en los datos. Saltando este análisis.")
            continue
            
        print(f"--- Análisis para: {var} ---")
        
        model_formula = f"{var} ~ C(ANO) + C(METODO_EXTRACCION) + C(ANO):C(METODO_EXTRACCION)"
        
        model = ols(model_formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print("\nTabla de ANOVA de Dos Vías:")
        print(anova_table)
        print("\n")

        df['INTERACTION'] = df['ANO'].astype(str) + "_" + df['METODO_EXTRACCION'].astype(str)
        tukey_results = pairwise_tukeyhsd(df[var], df['INTERACTION'], alpha=0.05)
        
        print("Resultados de la Prueba Post-Hoc de Tukey (HSD):")
        print(tukey_results)
        print("\n" + "="*80 + "\n")


    # --- 4. Análisis de Correlación de Pearson ---
    numeric_cols_for_corr = [
        'FOLIN_MG_EAG', 'DPPH_PORC', 'DPPH_MG_TROLOX', 
        'ABTS_PORC', 'ABTS_MG_TROLOX', 'FRAP_MG_TROLOX', 'FLAVONOIDES_MG_RUTINA'
    ]
    existing_cols = [col for col in numeric_cols_for_corr if col in df.columns]
    correlation_matrix = df[existing_cols].corr(method='pearson')

    print("--- Matriz de Correlación (Coeficiente de Pearson) ---")
    print(correlation_matrix)
    print("\n" + "="*80 + "\n")

# Restaurar la salida estándar original para ver los mensajes finales en la consola.
sys.stdout = original_stdout

# --- 5. Visualización y Guardado de la Matriz de Correlación ---
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True,         
    cmap='coolwarm',    
    fmt=".2f",          
    linewidths=.5
)
plt.title('Mapa de Calor de Correlación de Pearson entre Variables', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar la imagen en la carpeta de resultados.
correlation_img_path = os.path.join(output_dir, 'matriz_correlacion.png')
plt.savefig(correlation_img_path, dpi=300)

# El gráfico no se mostrará en pantalla para automatizar la ejecución,
# pero si quieres verlo, descomenta la siguiente línea:
# plt.show()

print(f"¡Análisis completado! Los resultados se han guardado en la carpeta: {output_dir}")

