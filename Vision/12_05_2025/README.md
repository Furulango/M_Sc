# Sistema de Clasificación de Fallas en Sistemas de Refrigeración

Este proyecto implementa un sistema de clasificación de fallas en sistemas de refrigeración basado en machine learning. Utiliza el algoritmo Random Forest para clasificar fallas en tres categorías: FALLA ALTA, FALLA BAJA y SANO, a partir de gradientes térmicos de tres componentes: válvula de expansión, compresor y condensador.

## Estructura del Proyecto

```
.
├── Generables/
│   ├── imagenes/           # Visualizaciones y gráficos generados
│   ├── modelos/            # Modelos entrenados
│   └── scripts/            # Scripts para predicción y visualización
│       ├── predecir_falla.py
│       └── app_clasificador_fallas.py
├── clasificador_fallas_refrigeracion.py   # Script principal
├── lanzador_herramientas.py               # Interfaz para acceder a las herramientas
└── README.md               # Este archivo
```

## Requisitos

El sistema requiere las siguientes bibliotecas de Python:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- tkinter (incluido en la mayoría de instalaciones de Python)

Puedes instalar todas las dependencias con:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Uso del Sistema

### 1. Script Principal

El script principal `clasificador_fallas_refrigeracion.py` realiza las siguientes funciones:

- Carga y preprocesa los datos desde un archivo CSV
- Entrena un modelo Random Forest para clasificar fallas
- Evalúa el modelo y genera visualizaciones
- Guarda el modelo entrenado y genera scripts de aplicación

Para ejecutar el análisis completo:

```bash
python clasificador_fallas_refrigeracion.py
```

### 2. Lanzador de Herramientas

Para facilitar el uso del sistema, se proporciona un lanzador de herramientas:

```bash
python lanzador_herramientas.py
```

Esta interfaz gráfica te permite:
- Ejecutar la aplicación de clasificación
- Ejecutar el predictor de consola
- Abrir la carpeta de imágenes generadas
- Explorar los archivos generados

### 3. Aplicación de Clasificación

La aplicación de clasificación proporciona una interfaz gráfica para realizar predicciones con el modelo entrenado:

```bash
python Generables/scripts/app_clasificador_fallas.py
```

Esta aplicación te permite:
- Ingresar valores de gradientes térmicos mediante controles deslizantes
- Visualizar las probabilidades de clasificación
- Ver gráficos de las predicciones en tiempo real

### 4. Función de Predicción

También puedes usar la función de predicción directamente desde tu código:

```python
from Generables.scripts.predecir_falla import predecir_falla

# Valores de ejemplo para los gradientes térmicos
valvula_expansion = 0.5
compresor = 0.8
condensador = 0.3

# Realizar predicción
resultado = predecir_falla(valvula_expansion, compresor, condensador)
print(f"Predicción: {resultado['prediccion']}")
print("\nProbabilidades:")
print(resultado['probabilidades'])
```

## Resultados y Visualizaciones

El sistema genera varias visualizaciones que se guardan en la carpeta `Generables/imagenes/`:

- **matriz_confusion_rf.png**: Matriz de confusión del modelo
- **importancia_caracteristicas_rf.png**: Importancia de cada característica
- **superficie_decision_rf.png**: Visualización de la superficie de decisión
- **curva_aprendizaje_rf.png**: Curva de aprendizaje del modelo
- **distribucion_clases.png**: Distribución de las clases en el conjunto de datos
- **pairplot_caracteristicas.png**: Relaciones entre pares de características
- **boxplots_caracteristicas.png**: Distribución de cada característica por clase
- **visualizacion_3d.png**: Visualización tridimensional de las características
- **heatmap_correlacion.png**: Correlación entre características
- **importancia_permutacion.png**: Importancia de características mediante permutación
- **curvas_roc.png**: Curvas ROC para evaluación del modelo

## Optimización de Hiperparámetros

El sistema incluye una funcionalidad para optimizar los hiperparámetros del modelo mediante GridSearchCV. Esta opción puede activarse durante la ejecución del script principal.

## Notas Adicionales

- El modelo está entrenado para clasificar en tres categorías: FALLA ALTA, FALLA BAJA y SANO.
- Se utilizan los gradientes térmicos de tres componentes: válvula de expansión, compresor y condensador.
- El modelo Random Forest proporciona tanto la clasificación como las probabilidades para cada clase.
- Todas las visualizaciones y resultados se guardan automáticamente en la carpeta `Generables`.

## Contacto

Para cualquier consulta o sugerencia sobre este sistema, por favor contacta al equipo de desarrollo.