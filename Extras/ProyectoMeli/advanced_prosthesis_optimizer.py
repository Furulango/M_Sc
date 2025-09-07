# Optimizador Simple y Robusto para Datos Grid Experimentales
# Enfoque directo sin interpolación compleja

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class SimpleGridOptimizer:
    """
    Optimizador simple y robusto específicamente para datos de grid experimental.
    Enfoque directo sin interpolaciones complejas que pueden fallar.
    """
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.targets = {
            'proximal': 1.21E+07,
            'medial': 1.55E+07,
            'distal': 3.88E+07
        }
        self.param_names = ['Rproximal', 'Rdistal', 'incremento', 'Dst']
        
    def preprocess_data(self):
        """Preprocesamiento simple y robusto"""
        print("=== ANÁLISIS DE DATOS EXPERIMENTALES ===")
        
        # Limpiar datos
        self.data = self.data.dropna(subset=self.param_names + 
                                   ['Esf_proximal', 'Esf_medial', 'Esf_distal'])
        
        self.X = self.data[self.param_names].values
        self.y = self.data[['Esf_proximal', 'Esf_medial', 'Esf_distal']].values
        
        print(f"Datos válidos: {len(self.data)}")
        
        # Análisis de la estructura del grid
        self.unique_values = {}
        for i, name in enumerate(self.param_names):
            unique_vals = np.unique(self.X[:, i])
            self.unique_values[name] = unique_vals
            print(f"{name}: {len(unique_vals)} valores únicos = {unique_vals}")
        
        return len(self.data)
    
    def method_1_direct_search(self):
        """Método 1: Búsqueda directa en puntos experimentales existentes"""
        print("\n=== MÉTODO 1: BÚSQUEDA DIRECTA EN DATOS EXPERIMENTALES ===")
        
        targets_array = np.array([self.targets['proximal'], self.targets['medial'], self.targets['distal']])
        
        best_idx = -1
        best_score = float('inf')
        best_params = None
        best_prediction = None
        
        print("Evaluando todos los puntos experimentales...")
        
        for i in range(len(self.X)):
            # Calcular error normalizado para cada punto experimental
            actual_stress = self.y[i]
            errors = [(actual_stress[j] - targets_array[j])**2 / targets_array[j]**2 
                     for j in range(3)]
            rmse_normalized = np.sqrt(np.mean(errors))
            
            if rmse_normalized < best_score:
                best_score = rmse_normalized
                best_idx = i
                best_params = self.X[i].copy()
                best_prediction = actual_stress.copy()
        
        print(f"Mejor punto experimental encontrado (índice {best_idx}):")
        for j, name in enumerate(self.param_names):
            print(f"  {name}: {best_params[j]}")
        
        return best_params, best_prediction, best_score, 'experimental'
    
    def method_2_random_forest_simple(self):
        """Método 2: Random Forest con evaluación en grid de valores únicos"""
        print("\n=== MÉTODO 2: RANDOM FOREST + GRID DE VALORES ÚNICOS ===")
        
        # Entrenar RF simple
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(self.X, self.y)
        
        # Evaluar rendimiento
        cv_scores = cross_val_score(rf, self.X, self.y, cv=5, scoring='neg_mean_squared_error')
        print(f"RF CV Score: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Generar todas las combinaciones posibles de valores únicos
        all_combinations = list(product(*[self.unique_values[name] for name in self.param_names]))
        print(f"Evaluando {len(all_combinations)} combinaciones de valores únicos...")
        
        targets_array = np.array([self.targets['proximal'], self.targets['medial'], self.targets['distal']])
        
        best_score = float('inf')
        best_params = None
        best_prediction = None
        
        for params in all_combinations:
            params_array = np.array(params).reshape(1, -1)
            
            # Predicción RF
            prediction = rf.predict(params_array)[0]
            
            # Calcular error normalizado
            errors = [(prediction[j] - targets_array[j])**2 / targets_array[j]**2 
                     for j in range(3)]
            rmse_normalized = np.sqrt(np.mean(errors))
            
            if rmse_normalized < best_score:
                best_score = rmse_normalized
                best_params = params
                best_prediction = prediction
        
        print(f"Mejor combinación encontrada:")
        for j, name in enumerate(self.param_names):
            print(f"  {name}: {best_params[j]}")
        
        return best_params, best_prediction, best_score, 'rf_grid'
    
    def method_3_weighted_average(self):
        """Método 3: Promedio ponderado de los mejores puntos experimentales"""
        print("\n=== MÉTODO 3: PROMEDIO PONDERADO DE MEJORES PUNTOS ===")
        
        targets_array = np.array([self.targets['proximal'], self.targets['medial'], self.targets['distal']])
        
        # Calcular errores para todos los puntos
        errors_and_indices = []
        for i in range(len(self.X)):
            actual_stress = self.y[i]
            errors = [(actual_stress[j] - targets_array[j])**2 / targets_array[j]**2 
                     for j in range(3)]
            rmse_normalized = np.sqrt(np.mean(errors))
            errors_and_indices.append((rmse_normalized, i))
        
        # Ordenar por error
        errors_and_indices.sort()
        
        # Tomar los mejores 10% de puntos
        n_best = max(3, len(self.X) // 10)
        best_indices = [idx for _, idx in errors_and_indices[:n_best]]
        
        print(f"Usando los mejores {n_best} puntos para promedio ponderado:")
        
        # Calcular pesos inversamente proporcionales al error
        weights = []
        for error, idx in errors_and_indices[:n_best]:
            weight = 1.0 / (error + 1e-10)  # Evitar división por cero
            weights.append(weight)
            print(f"  Punto {idx}: error={error:.4f}, peso={weight:.2f}")
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalizar
        
        # Promedio ponderado de parámetros
        best_params = np.average(self.X[best_indices], weights=weights, axis=0)
        
        # Promedio ponderado de predicciones (como estimación)
        best_prediction = np.average(self.y[best_indices], weights=weights, axis=0)
        
        # Calcular score del resultado promediado
        errors = [(best_prediction[j] - targets_array[j])**2 / targets_array[j]**2 
                 for j in range(3)]
        best_score = np.sqrt(np.mean(errors))
        
        print(f"Parámetros promedio ponderado:")
        for j, name in enumerate(self.param_names):
            print(f"  {name}: {best_params[j]:.4f}")
        
        return best_params, best_prediction, best_score, 'weighted_avg'
    
    def compare_methods(self):
        """Comparar los tres métodos y elegir el mejor"""
        print("\n" + "="*60)
        print("COMPARACIÓN DE MÉTODOS")
        print("="*60)
        
        # Ejecutar todos los métodos
        methods = [
            ("Búsqueda Directa", self.method_1_direct_search),
            ("Random Forest + Grid", self.method_2_random_forest_simple),
            ("Promedio Ponderado", self.method_3_weighted_average)
        ]
        
        results = []
        for method_name, method_func in methods:
            try:
                params, prediction, score, method_type = method_func()
                results.append({
                    'name': method_name,
                    'params': params,
                    'prediction': prediction,
                    'score': score,
                    'type': method_type,
                    'success': True
                })
            except Exception as e:
                print(f"Error en {method_name}: {e}")
                results.append({
                    'name': method_name,
                    'success': False,
                    'error': str(e)
                })
        
        # Encontrar el mejor método
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            print("ERROR: Todos los métodos fallaron")
            return None
        
        best_method = min(successful_results, key=lambda x: x['score'])
        
        print(f"\nRESULTADOS COMPARACIÓN:")
        for result in results:
            if result['success']:
                print(f"{result['name']}: Score = {result['score']:.4f}")
            else:
                print(f"{result['name']}: FALLÓ - {result['error']}")
        
        print(f"\nMEJOR MÉTODO: {best_method['name']} (Score: {best_method['score']:.4f})")
        
        return best_method
    
    def analyze_results(self, best_method):
        """Análisis detallado de los resultados"""
        print("\n" + "="*60)
        print("ANÁLISIS DETALLADO DE RESULTADOS")
        print("="*60)
        
        params = best_method['params']
        prediction = best_method['prediction']
        
        print(f"Método utilizado: {best_method['name']}")
        print(f"Tipo: {best_method['type']}")
        
        print(f"\nParámetros óptimos:")
        for i, name in enumerate(self.param_names):
            # Verificar si está dentro del rango experimental
            min_val = self.unique_values[name].min()
            max_val = self.unique_values[name].max()
            
            if best_method['type'] == 'experimental':
                status = "✓ EXPERIMENTAL"
            elif min_val <= params[i] <= max_val:
                status = "✓ INTERPOLACIÓN"
            else:
                status = "⚠️ EXTRAPOLACIÓN"
                
            print(f"  {name}: {params[i]:.4f} {status}")
        
        print(f"\nResultados vs objetivos:")
        targets_list = list(self.targets.values())
        region_names = ['Proximal', 'Medial', 'Distal']
        
        total_error = 0
        for i, region in enumerate(region_names):
            error_pct = abs(prediction[i] - targets_list[i]) / targets_list[i] * 100
            total_error += error_pct
            
            status = ("✓ EXCELENTE" if error_pct < 5 else 
                     "✓ BUENO" if error_pct < 10 else 
                     "⚠️ ACEPTABLE" if error_pct < 15 else 
                     "❌ REVISAR")
            
            print(f"  {region}: {prediction[i]/1e6:.2f}M vs {targets_list[i]/1e6:.2f}M")
            print(f"    Error: {error_pct:.1f}% - {status}")
        
        avg_error = total_error / 3
        print(f"\nError promedio: {avg_error:.1f}%")
        
        # Recomendación
        if best_method['type'] == 'experimental':
            reliability = "ALTA (punto medido experimentalmente)"
        elif avg_error < 10:
            reliability = "MEDIA-ALTA (interpolación confiable)"
        else:
            reliability = "MEDIA (requiere validación)"
        
        print(f"Confiabilidad: {reliability}")
        
        # Recomendación de acción
        if best_method['type'] == 'experimental' and avg_error < 12:
            recommendation = "IMPLEMENTAR - Usar estos parámetros para validación experimental"
        elif avg_error < 15:
            recommendation = "VALIDAR - Realizar experimento confirmatorio"
        else:
            recommendation = "REVISAR - Considerar experimentos adicionales"
        
        print(f"Recomendación: {recommendation}")
        
        return {
            'method': best_method['name'],
            'params': params,
            'prediction': prediction,
            'avg_error': avg_error,
            'recommendation': recommendation,
            'reliability': reliability
        }
    
    def run_optimization(self):
        """Ejecutar optimización completa simple y robusta"""
        print("OPTIMIZADOR SIMPLE Y ROBUSTO PARA DATOS GRID")
        print("="*60)
        
        # Preprocesar datos
        n_data = self.preprocess_data()
        
        if n_data < 10:
            print("ERROR: Datos insuficientes para optimización")
            return None
        
        # Comparar métodos
        best_method = self.compare_methods()
        
        if best_method is None:
            return None
        
        # Análisis detallado
        analysis = self.analyze_results(best_method)
        
        print("\n" + "="*60)
        print("RESUMEN FINAL")
        print("="*60)
        print(f"Mejor método: {analysis['method']}")
        print(f"Error promedio: {analysis['avg_error']:.1f}%")
        print(f"Confiabilidad: {analysis['reliability']}")
        print(f"Recomendación: {analysis['recommendation']}")
        
        return analysis

# EJECUTAR OPTIMIZACIÓN SIMPLE
if __name__ == "__main__":
    print("OPTIMIZADOR SIMPLE PARA DATOS GRID EXPERIMENTALES")
    print("Sin interpolaciones complejas - Solo métodos robustos\n")

    optimizer = SimpleGridOptimizer('D:\\GitHub\\M_Sc\\Extras\\ProyectoMeli\\Base_datos_protesis.csv')
    results = optimizer.run_optimization()
    
    if results:
        print(f"\nPARAMETROS ÓPTIMOS FINALES:")
        param_names = ['Rproximal', 'Rdistal', 'incremento', 'Dst']
        for i, name in enumerate(param_names):
            print(f"{name}: {results['params'][i]:.4f}")
    else:
        print("OPTIMIZACIÓN FALLÓ - Revisar datos de entrada")