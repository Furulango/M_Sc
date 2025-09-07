Plan de Mejoras para el Optimizador de Carteras
Este documento detalla las próximas tareas para mejorar el rendimiento, la robustez y la inteligencia del modelo de optimización de carteras. Las tareas se dividen en tres enfoques estratégicos.

Enfoque 1: Implementación de Controles de Riesgo (Gestor de Riesgos)
Objetivo: Reducir la volatilidad y el Max Drawdown forzando la diversificación. Se espera una mejora directa en el Sharpe Ratio al disminuir el riesgo asumido.

[ ] Modificar el Algoritmo Genético para añadir restricciones de peso:

[ ] Dentro de la función genetic_algorithm_optimize, localizar el bucle de generación de la nueva población (while len(new_population) < population_size:).

[ ] Después del paso de mutación y antes de la normalización, introducir una variable max_weight_limit (ej. 0.30 para un límite del 30%).

[ ] Aplicar np.clip(child, 0, max_weight_limit) a cada nueva solución (child) para asegurar que ningún activo supere el umbral establecido.

[ ] Asegurarse de que la normalización posterior (child / child.sum()) siga funcionando correctamente.

[ ] Realizar un nuevo backtest y comparar métricas:

[ ] Ejecutar el script con las nuevas restricciones.

[ ] Documentar la nueva Volatilidad Anual, Max Drawdown y Sharpe Ratio.

[ ] Comparar con los resultados originales para cuantificar la mejora en la gestión del riesgo.

Enfoque 2: Redefinición del Objetivo de Optimización (Ingeniero de ML)
Objetivo: Ajustar la función de fitness del algoritmo para que penalice de forma más agresiva la concentración o para que optimice una métrica de riesgo más sofisticada.

[ ] Aumentar la penalización por concentración en la función de fitness:

[ ] Dentro de la función genetic_algorithm_optimize, en el bucle de evaluación de fitness_scores.

[ ] Localizar la línea adjusted_sharpe = sharpe - 0.3 * concentration_penalty.

[ ] Incrementar el factor de penalización de 0.3 a un valor superior (ej. 0.8 o 1.0) para desincentivar más fuertemente las carteras concentradas.

[ ] Ejecutar y analizar si esta modificación conduce a carteras más diversificadas y a un mejor Sharpe Ratio.

[ ] (Opcional - Avanzado) Investigar la implementación del Sortino Ratio:

[ ] Estudiar la fórmula del Sortino Ratio, que requiere el cálculo de la desviación a la baja (downside deviation).

[ ] Modificar la función de fitness para calcular y retornar el Sortino Ratio en lugar del Sharpe Ratio. Esto requerirá acceso a la serie de retornos históricos de la cartera dentro del bucle de evaluación.

Enfoque 3: Enriquecimiento de Datos y Modelo Predictivo (Científico de Datos)
Objetivo: Mejorar la precisión del "cerebro" del sistema (el modelo LSTM) alimentándolo con más información relevante para que sus predicciones de volatilidad sean más acertadas.

[ ] Ampliar la adquisición de datos:

[ ] En la función download_data, modificar la llamada a yf.Ticker.history para descargar no solo el precio de cierre (Close), sino también Volume, High y Low.

[ ] Implementar Feature Engineering:

[ ] Crear una función de preprocesamiento que, para cada activo, calcule nuevas características (features) a partir de los datos brutos.

[ ] Features a añadir: Media móvil de 50 días, Media móvil de 200 días, RSI (Relative Strength Index), retorno diario, volatilidad histórica.

[ ] Convertir el LSTM en un modelo multivariado:

[ ] Modificar la función lstm_predict_volatility para que acepte un DataFrame con múltiples columnas (features) como entrada.

[ ] Ajustar el MinMaxScaler para que escale todas las features juntas.

[ ] Reestructurar la creación de las secuencias de datos (X, y) para que X contenga secuencias de todas las features, manteniendo y como la volatilidad a predecir.

[ ] Actualizar la input_shape de la primera capa LSTM en el modelo Keras para que acepte el nuevo número de features: input_shape=(lookback_period, num_features).

[ ] Integrar y probar el nuevo modelo predictivo:

[ ] Asegurarse de que el backtest llame correctamente a la nueva función LSTM multivariada.

[ ] Ejecutar la simulación completa y evaluar si la inteligencia adicional se traduce en un rendimiento superior de la cartera.

Ruta de Implementación Recomendada
Prioridad 1 (Base): Completar el Enfoque 1. Establece una base robusta y segura, lo cual es fundamental.

Prioridad 2 (Avance Core): Abordar el Enfoque 3. Esta es la mejora más significativa en términos de inteligencia del modelo y la que tiene mayor impacto para un portafolio de Machine Learning Engineer.

Prioridad 3 (Refinamiento): Considerar el Enfoque 2 como un ajuste final una vez que los otros dos estén implementados y validados.