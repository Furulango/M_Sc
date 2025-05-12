import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os

# Crear las carpetas necesarias
def crear_estructura_carpetas():
    """
    Crea la estructura de carpetas necesaria para organizar los resultados.
    """
    # Crear carpeta principal Generables
    os.makedirs('Generables', exist_ok=True)
    
    # Crear subcarpetas dentro de Generables
    os.makedirs('Generables/scripts', exist_ok=True)
    os.makedirs('Generables/imagenes', exist_ok=True)
    os.makedirs('Generables/modelos', exist_ok=True)
    
    print("Estructura de carpetas creada:")
    print("- Generables/")
    print("  |- scripts/")
    print("  |- imagenes/")
    print("  |- modelos/")
    
# Ruta al archivo CSV de gradientes
ruta_csv = r"D:\GitHub\M_Sc\Vision\Material\IMAGENES TERMICAS\gradientes_.csv"

# 1. Cargar los datos
def cargar_datos(ruta):
    """
    Carga los datos del archivo CSV y los prepara para el análisis.
    """
    try:
        # Cargar el archivo CSV
        print(f"Cargando datos desde: {ruta}")
        df = pd.read_csv(ruta)
        
        # Mostrar las primeras filas para verificar la estructura
        print("\nPrimeras filas del dataset:")
        print(df.head())
        
        # Información básica del dataset
        print("\nInformación del dataset:")
        print(f"Tamaño: {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return None

# 2. Preprocesamiento de datos
def preparar_datos(df):
    """
    Prepara los datos para el entrenamiento del modelo.
    Separa características (X) y etiquetas (y).
    """
    if df is None:
        return None, None, None, None
    
    # Verificar si hay valores faltantes
    if df.isnull().sum().any():
        print("\nSe encontraron valores faltantes. Realizando imputación...")
        df = df.fillna(df.mean())  # Imputar con la media
    
    # Asumiendo que la columna 'Carpeta' contiene las etiquetas (clases)
    # y el resto de columnas son características
    if 'Carpeta' in df.columns:
        # Convertir etiquetas categóricas a numéricas
        etiquetas_unicas = df['Carpeta'].unique()
        print(f"\nClases encontradas: {etiquetas_unicas}")
        
        # Crear un mapeo de etiquetas a números
        mapeo_etiquetas = {etiqueta: i for i, etiqueta in enumerate(etiquetas_unicas)}
        print(f"Mapeo de etiquetas: {mapeo_etiquetas}")
        
        # Convertir etiquetas a números
        y = df['Carpeta'].map(mapeo_etiquetas)
        
        # Seleccionar columnas para características (excluyendo 'Carpeta' e 'Imagen')
        columnas_caracteristicas = [col for col in df.columns if col not in ['Carpeta', 'Imagen']]
        X = df[columnas_caracteristicas]
        
        print(f"\nCaracterísticas seleccionadas: {columnas_caracteristicas}")
        print(f"Forma de X: {X.shape}")
        print(f"Forma de y: {y.shape}")
        
        # Normalizar las características (opcional pero recomendado)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y.values, mapeo_etiquetas, columnas_caracteristicas
    else:
        print("Error: No se encontró la columna 'Carpeta' para las etiquetas.")
        return None, None, None, None

# 3. Entrenamiento y evaluación del modelo Random Forest
def entrenar_evaluar_random_forest(X, y, mapeo_etiquetas, columnas_caracteristicas):
    """
    Entrena un modelo Random Forest y evalúa su rendimiento.
    """
    if X is None or y is None:
        return None, None
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\nDivisión de datos: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
    
    # Entrenar el modelo Random Forest
    print("\nEntrenando modelo Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = rf_clf.predict(X_test)
    
    # Calcular y mostrar métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión del modelo: {accuracy:.4f}")
    
    # Mapeo inverso para mostrar nombres de clases
    mapeo_inverso = {v: k for k, v in mapeo_etiquetas.items()}
    etiquetas = [mapeo_inverso[i] for i in range(len(mapeo_etiquetas))]
    
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred, target_names=etiquetas))
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.title('Matriz de Confusión', fontsize=14)
    plt.tight_layout()
    plt.savefig('Generables/imagenes/matriz_confusion_rf.png')
    
    # Importancia de características
    plt.figure(figsize=(12, 6))
    importancias = pd.Series(rf_clf.feature_importances_, index=columnas_caracteristicas)
    importancias = importancias.sort_values(ascending=False)
    ax = importancias.plot(kind='bar', colormap='viridis')
    plt.title('Importancia de Características', fontsize=14)
    plt.xlabel('Característica', fontsize=12)
    plt.ylabel('Importancia', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Añadir valores encima de cada barra
    for i, v in enumerate(importancias):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
    plt.savefig('Generables/imagenes/importancia_caracteristicas_rf.png')
    
    print("\nImportancia de características:")
    for feat, imp in zip(importancias.index, importancias.values):
        print(f"{feat}: {imp:.4f}")
    
    # Visualización de la superficie de decisión (para 2 características principales)
    if X.shape[1] >= 2:
        # Tomar las dos características más importantes
        top_features_idx = importancias.index[:2]
        
        # Crear un malla para visualizar la superficie de decisión
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # Para predecir en la malla, necesitamos preparar datos con todas las características
        # Para las características no usadas, usamos la media
        X_mean = X.mean(axis=0)
        n_features = X.shape[1]
        meshgrid_features = np.zeros((xx.ravel().shape[0], n_features))
        
        # Llenar con valores medios
        for i in range(n_features):
            meshgrid_features[:, i] = X_mean[i]
        
        # Reemplazar las dos características principales con los valores de la malla
        meshgrid_features[:, 0] = xx.ravel()
        meshgrid_features[:, 1] = yy.ravel()
        
        # Predecir
        Z = rf_clf.predict(meshgrid_features)
        Z = Z.reshape(xx.shape)
        
        # Graficar la superficie de decisión
        plt.figure(figsize=(12, 10))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        
        # Graficar los puntos
        for i, label in enumerate(np.unique(y)):
            idx = y == label
            plt.scatter(X[idx, 0], X[idx, 1], 
                        label=mapeo_inverso[label],
                        edgecolor='black', alpha=0.7)
            
        plt.xlabel(f'{top_features_idx[0]}', fontsize=12)
        plt.ylabel(f'{top_features_idx[1]}', fontsize=12)
        plt.title('Superficie de Decisión del Clasificador Random Forest', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig('Generables/imagenes/superficie_decision_rf.png')
    
    # Curva de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42), 
        X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Calcular medias y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Graficar curva de aprendizaje
    plt.figure(figsize=(12, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Puntuación de entrenamiento")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Puntuación de validación cruzada")
    plt.title("Curva de Aprendizaje para Random Forest", fontsize=14)
    plt.xlabel("Número de muestras de entrenamiento", fontsize=12)
    plt.ylabel("Puntuación", fontsize=12)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('Generables/imagenes/curva_aprendizaje_rf.png')
    
    # Validación cruzada
    cv_scores = cross_val_score(rf_clf, X, y, cv=5)
    print("\nResultados de Validación Cruzada (5-fold):")
    print(f"Puntuaciones individuales: {cv_scores}")
    print(f"Puntuación media: {cv_scores.mean():.4f}")
    print(f"Desviación estándar: {cv_scores.std():.4f}")
    
    return rf_clf, mapeo_inverso

# 4. Optimización de hiperparámetros (opcional)
def optimizar_hiperparametros(X, y):
    """
    Realiza una búsqueda de cuadrícula para encontrar los mejores hiperparámetros.
    """
    if X is None or y is None:
        return None
    
    print("\nOptimizando hiperparámetros con GridSearchCV...")
    
    # Definir parámetros para la búsqueda
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Crear el modelo base
    rf = RandomForestClassifier(random_state=42)
    
    # Configurar la búsqueda de cuadrícula con validación cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    
    # Realizar la búsqueda
    grid_search.fit(X, y)
    
    # Mostrar los mejores parámetros
    print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor puntuación: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 5. Visualización de resultados
def visualizar_resultados(df, mapeo_etiquetas):
    """
    Crea visualizaciones de los datos y resultados.
    """
    if df is None:
        return
    
    # Visualizar distribución de clases
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Carpeta', data=df, palette='viridis')
    plt.title('Distribución de Clases', fontsize=14)
    plt.xlabel('Tipo de Falla', fontsize=12)
    plt.ylabel('Número de Muestras', fontsize=12)
    
    # Añadir etiquetas de conteo encima de cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', fontsize=12)
    
    plt.xticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig('Generables/imagenes/distribucion_clases.png')
    
    # Visualizar las características por pares para cada clase
    plt.figure(figsize=(16, 12))
    sns.pairplot(df, hue='Carpeta', vars=['Valvula_de_expansion', 'Compresor', 'Condensador'], 
                palette='viridis', diag_kind='kde', markers=['o', 's', 'D'])
    plt.suptitle('Visualización de Pares de Características por Clase', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('Generables/imagenes/pairplot_caracteristicas.png')
    
    # Visualizar distribución de cada característica por clase
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, feature in enumerate(['Valvula_de_expansion', 'Compresor', 'Condensador']):
        sns.boxplot(x='Carpeta', y=feature, data=df, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Distribución de {feature}', fontsize=14)
        axes[i].set_xlabel('Tipo de Falla', fontsize=12)
        axes[i].set_ylabel(f'Valor de {feature}', fontsize=12)
        axes[i].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('Generables/imagenes/boxplots_caracteristicas.png')
    
    # Gráfico 3D para visualizar todas las características juntas
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colores para cada clase
    colores = {'FALLA ALTA': 'red', 'FALLA BAJA': 'orange', 'SANO': 'green'}
    
    for clase in df['Carpeta'].unique():
        subset = df[df['Carpeta'] == clase]
        ax.scatter(subset['Valvula_de_expansion'], 
                  subset['Compresor'], 
                  subset['Condensador'],
                  c=colores[clase], label=clase, alpha=0.7)
    
    ax.set_xlabel('Válvula de Expansión', fontsize=12)
    ax.set_ylabel('Compresor', fontsize=12)
    ax.set_zlabel('Condensador', fontsize=12)
    ax.set_title('Visualización 3D de las Características', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Generables/imagenes/visualizacion_3d.png')
    
    # Mapa de calor de correlación
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['Valvula_de_expansion', 'Compresor', 'Condensador']].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                mask=mask, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Mapa de Calor de Correlación entre Características', fontsize=14)
    plt.tight_layout()
    plt.savefig('Generables/imagenes/heatmap_correlacion.png')

# Función para generar curvas ROC
def generar_curvas_roc(modelo, X, y, mapeo_inverso):
    """
    Genera curvas ROC para un problema de clasificación multiclase.
    """
    # Dividir los datos para evaluación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # Obtener probabilidades para cada clase
    y_probs = modelo.predict_proba(X_test)
    
    # Número de clases
    n_classes = len(np.unique(y))
    
    # Calcular curva ROC para cada clase
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        # Calcular ROC para la clase actual (one-vs-rest)
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Graficar curva ROC
        plt.plot(fpr, tpr, lw=2,
                 label=f'ROC para clase {mapeo_inverso[i]} (AUC = {roc_auc:.2f})')
    
    # Línea diagonal de referencia
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC para Clasificación Multiclase', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Generables/imagenes/curvas_roc.png')

# Función para crear script de predicción
def crear_funcion_prediccion(mapeo_inverso):
    """
    Crea un script de Python para hacer predicciones con el modelo guardado.
    """
    codigo = """
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
    print("\\nProbabilidades:")
    print(resultado['probabilidades'])
"""
    
    # Guardar el código en un archivo
    ruta_script = 'Generables/scripts/predecir_falla.py'
    with open(ruta_script, 'w') as f:
        f.write(codigo)
    
    print(f"Función de predicción guardada en '{ruta_script}'")
    print("Puedes usar esta función para hacer predicciones con nuevos datos de gradientes.")

# Función para crear la aplicación GUI
def generar_app_gui():
    """
    Genera una aplicación GUI sencilla para visualizar y probar el modelo.
    """
    codigo_gui = """
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import pandas as pd
import seaborn as sns

class AppClasificadorFallas:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Fallas en Sistemas de Refrigeración")
        self.root.geometry("900x680")
        self.root.configure(bg="#f0f0f0")
        
        # Ruta relativa a la carpeta de modelos
        carpeta_actual = os.path.dirname(os.path.abspath(__file__))
        carpeta_base = os.path.dirname(os.path.dirname(carpeta_actual))
        modelo_path = os.path.join(carpeta_base, 'Generables', 'modelos', 'random_forest_model.pkl')
        
        if not os.path.exists(modelo_path):
            messagebox.showerror("Error", f"Modelo no encontrado en: {modelo_path}")
            self.root.destroy()
            return
            
        self.modelo = joblib.load(modelo_path)
        self.mapeo_inverso = {0: 'FALLA ALTA', 1: 'FALLA BAJA', 2: 'SANO'}
        
        # Título principal
        titulo = tk.Label(root, text="Sistema de Clasificación de Fallas", 
                         font=("Arial", 18, "bold"), bg="#f0f0f0", pady=10)
        titulo.pack(fill="x")
        
        # Frame para entradas
        frame_entrada = ttk.LabelFrame(root, text="Datos de Gradientes Térmicos")
        frame_entrada.pack(fill="both", expand=False, padx=20, pady=10)
        
        # Crear widgets para entrada de datos
        self.crear_widgets_entrada(frame_entrada)
        
        # Frame para resultados
        self.frame_resultados = ttk.LabelFrame(root, text="Resultados de la Clasificación")
        self.frame_resultados.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Inicializar frame de gráficos
        self.frame_grafico = ttk.Frame(self.frame_resultados)
        self.frame_grafico.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=10)
        
        # Texto inicial en el área de resultados
        self.resultado_text = tk.Label(self.frame_resultados, 
                                      text="Los resultados aparecerán aquí",
                                      font=("Arial", 12), pady=10)
        self.resultado_text.pack()
        
        # Inicializar figura vacía
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_grafico)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Botones de acción
        frame_botones = ttk.Frame(root)
        frame_botones.pack(fill="x", padx=20, pady=10)
        
        # Botón para predecir
        btn_predecir = ttk.Button(frame_botones, text="Predecir", command=self.predecir)
        btn_predecir.pack(side=tk.LEFT, padx=10)
        
        # Botón para limpiar
        btn_limpiar = ttk.Button(frame_botones, text="Limpiar", command=self.limpiar)
        btn_limpiar.pack(side=tk.LEFT, padx=10)
        
        # Botón para salir
        btn_salir = ttk.Button(frame_botones, text="Salir", command=root.destroy)
        btn_salir.pack(side=tk.RIGHT, padx=10)
    
    def crear_widgets_entrada(self, frame):
        # Crear 3 entradas para los valores de gradiente
        labels = ["Válvula de Expansión:", "Compresor:", "Condensador:"]
        self.entradas = []
        
        for i, label in enumerate(labels):
            # Frame para cada entrada
            frame_entrada = ttk.Frame(frame)
            frame_entrada.pack(fill="x", padx=10, pady=5)
            
            # Etiqueta
            lbl = ttk.Label(frame_entrada, text=label, width=20)
            lbl.pack(side=tk.LEFT, padx=5)
            
            # Campo de entrada
            var = tk.DoubleVar(value=0.0)
            entrada = ttk.Entry(frame_entrada, textvariable=var, width=10)
            entrada.pack(side=tk.LEFT, padx=5)
            
            # Slider
            slider = ttk.Scale(frame_entrada, from_=-5.0, to=5.0, 
                              orient="horizontal", variable=var, 
                              length=400)
            slider.pack(side=tk.LEFT, fill="x", expand=True, padx=10)
            
            self.entradas.append(var)
    
    def predecir(self):
        try:
            # Obtener valores de las entradas
            valvula = self.entradas[0].get()
            compresor = self.entradas[1].get()
            condensador = self.entradas[2].get()
            
            # Preparar datos para predicción
            X = np.array([[valvula, compresor, condensador]])
            
            # Realizar predicción
            y_pred = self.modelo.predict(X)[0]
            y_probs = self.modelo.predict_proba(X)[0]
            
            # Obtener clase predicha
            clase = self.mapeo_inverso[y_pred]
            
            # Actualizar texto de resultado
            resultado = f"Clasificación: {clase}\n\n"
            resultado += "Probabilidades:\n"
            
            for i, prob in enumerate(y_probs):
                resultado += f"{self.mapeo_inverso[i]}: {prob:.4f}\n"
            
            self.resultado_text.config(text=resultado)
            
            # Crear gráfico de barras para las probabilidades
            self.ax.clear()
            clases = [self.mapeo_inverso[i] for i in range(len(y_probs))]
            colors = ['red' if clase == 'FALLA ALTA' else 'orange' if clase == 'FALLA BAJA' else 'green' 
                    for clase in clases]
            
            bars = self.ax.bar(clases, y_probs, color=colors)
            self.ax.set_ylim(0, 1)
            self.ax.set_ylabel('Probabilidad')
            self.ax.set_title('Probabilidades de Predicción')
            
            # Añadir etiquetas con los valores
            for bar in bars:
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {str(e)}")
    
    def limpiar(self):
        # Restaurar valores predeterminados
        for var in self.entradas:
            var.set(0.0)
        
        # Limpiar resultados
        self.resultado_text.config(text="Los resultados aparecerán aquí")
        
        # Limpiar gráfico
        self.ax.clear()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppClasificadorFallas(root)
    root.mainloop()
"""
    
    # Guardar el código GUI en un archivo
    ruta_gui = 'Generables/scripts/app_clasificador_fallas.py'
    with open(ruta_gui, 'w') as f:
        f.write(codigo_gui)
    
    print(f"Aplicación GUI guardada en '{ruta_gui}'")
    print("Puedes ejecutar esta aplicación para visualizar y probar el modelo interactivamente.")

# Función principal
def main():
    # Crear estructura de carpetas
    crear_estructura_carpetas()
    
    # 1. Cargar datos
    df = cargar_datos(ruta_csv)
    
    if df is not None:
        # 2. Visualizar datos originales
        visualizar_resultados(df, None)
        
        # 3. Preparar datos
        X, y, mapeo_etiquetas, columnas_caracteristicas = preparar_datos(df)
        
        if X is not None and y is not None:
            # 4. Entrenar y evaluar modelo básico
            rf_model, mapeo_inverso = entrenar_evaluar_random_forest(X, y, mapeo_etiquetas, columnas_caracteristicas)
            
            # Guardar el modelo entrenado
            joblib.dump(rf_model, 'Generables/modelos/random_forest_model.pkl')
            print("\nModelo guardado en 'Generables/modelos/random_forest_model.pkl'")
            
            # Crear función de predicción para nuevos datos
            crear_funcion_prediccion(mapeo_inverso)
            
            # Crear una aplicación GUI para probar el modelo
            generar_app_gui()
            
            # 5. Optimización de hiperparámetros (opcional, puede tardar)
            realizar_optimizacion = input("\n¿Desea realizar optimización de hiperparámetros? (s/n): ").lower() == 's'
            if realizar_optimizacion:
                mejor_modelo = optimizar_hiperparametros(X, y)
                
                # Evaluar el modelo optimizado
                if mejor_modelo is not None:
                    print("\nEvaluando modelo optimizado...")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                    mejor_modelo.fit(X_train, y_train)
                    y_pred = mejor_modelo.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"\nPrecisión del modelo optimizado: {accuracy:.4f}")
                    
                    etiquetas = [mapeo_inverso[i] for i in range(len(mapeo_etiquetas))]
                    print("\nInforme de clasificación del modelo optimizado:")
                    print(classification_report(y_test, y_pred, target_names=etiquetas))
                    
                    # Guardar modelo optimizado
                    joblib.dump(mejor_modelo, 'Generables/modelos/random_forest_optimizado.pkl')
                    print("Modelo optimizado guardado en 'Generables/modelos/random_forest_optimizado.pkl'")
            
            # 6. Evaluar modelo con validación cruzada
            print("\nRealizando validación cruzada...")
            cv_scores = cross_val_score(rf_model, X, y, cv=5)
            print(f"Puntuaciones de validación cruzada: {cv_scores}")
            print(f"Puntuación media de validación cruzada: {cv_scores.mean():.4f}")
            print(f"Desviación estándar: {cv_scores.std():.4f}")
            
            # 7. Analizar el impacto de cada característica
            print("\nCalculando importancia de características mediante permutación...")
            perm_importance = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)
            
            features_importance = pd.DataFrame({
                'Característica': columnas_caracteristicas,
                'Importancia': perm_importance.importances_mean,
                'STD': perm_importance.importances_std
            }).sort_values('Importancia', ascending=False)
            
            print("\nImportancia mediante permutación:")
            print(features_importance)
            
            # Graficar importancia mediante permutación
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Característica', y='Importancia', data=features_importance)
            plt.errorbar(x=range(len(features_importance)), 
                        y=features_importance['Importancia'], 
                        yerr=features_importance['STD'], fmt='none', c='black')
            plt.title('Importancia de Características mediante Permutación', fontsize=14)
            plt.xlabel('Característica', fontsize=12)
            plt.ylabel('Disminución en Precisión', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('Generables/imagenes/importancia_permutacion.png')
            
            # 8. Generar curvas ROC para análisis multiclase
            generar_curvas_roc(rf_model, X, y, mapeo_inverso)
            
            print("\nAnálisis completado exitosamente. Todos los resultados y visualizaciones han sido guardados.")
            print("\nArchivos generados en la carpeta 'Generables':")
            print("\nImagenes generadas:")
            for archivo in os.listdir('Generables/imagenes'):
                if archivo.endswith('.png'):
                    print(f"- imagenes/{archivo}")
            
            print("\nScripts generados:")
            for archivo in os.listdir('Generables/scripts'):
                if archivo.endswith('.py'):
                    print(f"- scripts/{archivo}")
                
            print("\nModelos guardados:")
            for archivo in os.listdir('Generables/modelos'):
                if archivo.endswith('.pkl'):
                    print(f"- modelos/{archivo}")
            
            print("\nPuedes ejecutar la aplicación GUI con el comando: python Generables/scripts/app_clasificador_fallas.py")
        else:
            print("No se pudo preparar los datos para el análisis.")
    else:
        print("No se pudo cargar el archivo CSV.")

if __name__ == "__main__":
    main()