import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns

class ProsthesisOptimizer:
    def __init__(self, data_path):
        """
        Optimizador - prótesis de cadera
        """
        self.data = pd.read_csv(data_path)
        self.targets = {
            'proximal': 1.21E+07,
            'medial': 1.55E+07,
            'distal': 3.88E+07
        }
        self.models = {}
        self.scalers = {}
        
    def preprocess_data(self):
        """Limpieza y preparación de datos"""
        self.data = self.data.dropna(subset=['Rproximal', 'Rdistal', 'incremento', 'Dst',
                                           'Esf_proximal', 'Esf_medial', 'Esf_distal'])
        self.X = self.data[['Rproximal', 'Rdistal', 'incremento', 'Dst']].values
        self.y = self.data[['Esf_proximal', 'Esf_medial', 'Esf_distal']].values
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(self.X)
        self.y_scaled = self.scaler_y.fit_transform(self.y)
        print(f"Datos procesados: {len(self.data)} muestras")
        print(f"Rangos de entrada:")
        for i, col in enumerate(['Rproximal', 'Rdistal', 'incremento', 'Dst']):
            print(f"  {col}: [{self.X[:, i].min():.2f}, {self.X[:, i].max():.2f}]")
    
    def compare_models(self):
        """Comparar modelos de regresión"""
        models = {
            'Gaussian Process': GaussianProcessRegressor(
                kernel=ConstantKernel() * RBF() + ConstantKernel(),
                alpha=1e-10,
                n_restarts_optimizer=10
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        results = {}
        for name, model in models.items():
            scores = []
            for i in range(3):
                cv_scores = cross_val_score(
                    model, self.X_scaled, self.y_scaled[:, i], 
                    cv=5, scoring='neg_mean_squared_error'
                )
                scores.append(-cv_scores.mean())
            results[name] = {
                'cv_mse_proximal': scores[0],
                'cv_mse_medial': scores[1], 
                'cv_mse_distal': scores[2],
                'cv_mse_avg': np.mean(scores)
            }
        print("\n=== COMPARACIÓN DE MODELOS ===")
        for name, scores in results.items():
            print(f"{name}:")
            print(f"  MSE Promedio CV: {scores['cv_mse_avg']:.4f}")
            print(f"  MSE por variable: {scores['cv_mse_proximal']:.4f}, {scores['cv_mse_medial']:.4f}, {scores['cv_mse_distal']:.4f}")
        return results
    
    def train_best_model(self, model_type='gp'):
        """Entrenar mejor modelo """
        if model_type == 'gp':
            self.model = GaussianProcessRegressor(
                kernel=ConstantKernel() * Matern(length_scale=1.0, nu=2.5),
                alpha=1e-6,
                n_restarts_optimizer=20
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        self.model.fit(self.X_scaled, self.y_scaled)
        print(f"Modelo {model_type} entrenado exitosamente")
    
    def objective_function(self, x):
        """Función objetivo optimización"""
        x = np.clip(x, 
                   [self.X[:, i].min() for i in range(4)],
                   [self.X[:, i].max() for i in range(4)])
        x_scaled = self.scaler_X.transform(x.reshape(1, -1))
        y_pred_scaled = self.model.predict(x_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
        errors = [(y_pred[i] - list(self.targets.values())[i])**2 
                 for i in range(3)]
        return np.sqrt(np.mean(errors))
    
    def optimize_design(self, method='differential_evolution'):
        """Optimizar diseño """
        bounds = [
            (self.X[:, 0].min(), self.X[:, 0].max()),
            (self.X[:, 1].min(), self.X[:, 1].max()),
            (self.X[:, 2].min(), self.X[:, 2].max()),
            (self.X[:, 3].min(), self.X[:, 3].max()),
        ]
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=1000,
                popsize=30
            )
        else:
            best_result = None
            best_score = float('inf')
            for _ in range(20):
                x0 = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
                result = minimize(
                    self.objective_function,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
            result = best_result
        return result
    
    def analyze_results(self, result):
        """Analizar resultados """
        optimal_params = result.x
        x_opt_scaled = self.scaler_X.transform(optimal_params.reshape(1, -1))
        y_opt_scaled = self.model.predict(x_opt_scaled)
        y_opt = self.scaler_y.inverse_transform(y_opt_scaled.reshape(1, -1))[0]
        print("\n=== RESULTADOS DE OPTIMIZACIÓN ===")
        print(f"Parámetros óptimos encontrados:")
        print(f"  Rproximal: {optimal_params[0]:.2f}")
        print(f"  Rdistal: {optimal_params[1]:.2f}")
        print(f"  incremento: {optimal_params[2]:.4f}")
        print(f"  Dst: {optimal_params[3]:.2f}")
        print(f"\nEsfuerzos predichos vs. objetivos:")
        targets_list = list(self.targets.values())
        for i, name in enumerate(['Proximal', 'Medial', 'Distal']):
            error_pct = abs(y_opt[i] - targets_list[i]) / targets_list[i] * 100
            print(f"  {name}: {y_opt[i]/1e6:.2f}M vs {targets_list[i]/1e6:.2f}M (Error: {error_pct:.1f}%)")
        print(f"\nRMSE total: {result.fun/1e6:.2f}M Pa")
        return optimal_params, y_opt
    
    def validate_solution(self, optimal_params):
        """Validar la solución """
        in_bounds = True
        param_names = ['Rproximal', 'Rdistal', 'incremento', 'Dst']
        for i, (param, name) in enumerate(zip(optimal_params, param_names)):
            min_val, max_val = self.X[:, i].min(), self.X[:, i].max()
            if param < min_val or param > max_val:
                print(f" {name} = {param:.2f} fuera del rango experimental [{min_val:.2f}, {max_val:.2f}]")
                in_bounds = False
        if in_bounds:
            print(" Solución está dentro del espacio experimental")
        return in_bounds
    
    def compare_optimization_results(self):
        """Comparar resultados entre GP y RF"""
        print("\n" + "="*60)
        print("COMPARACIÓN OPTIMIZACIÓN GP vs RF")
        print("="*60)
        results_comparison = {}
        for model_name in ['gp', 'rf']:
            print(f"\n--- OPTIMIZANDO CON {model_name.upper()} ---")
            self.train_best_model(model_name)
            result = self.optimize_design('differential_evolution')
            optimal_params, predicted_stress = self.analyze_results(result)
            in_bounds = self.validate_solution(optimal_params)
            targets_list = list(self.targets.values())
            errors = [abs(predicted_stress[i] - targets_list[i]) / targets_list[i] * 100 for i in range(3)]
            avg_error = np.mean(errors)
            results_comparison[model_name] = {
                'params': optimal_params,
                'prediction': predicted_stress,
                'rmse': result.fun,
                'avg_error': avg_error,
                'in_bounds': in_bounds,
                'individual_errors': errors
            }
        print("\n" + "="*60)
        print("COMPARATIVA")
        print("="*60)
        param_names = ['Rproximal', 'Rdistal', 'incremento', 'Dst']
        print("\nPARÁMETROS ÓPTIMOS:")
        print("-" * 50)
        print(f"{'Parámetro':<12} {'GP':<12} {'RF':<12} {'Diferencia':<12}")
        print("-" * 50)
        for i, name in enumerate(param_names):
            gp_val = results_comparison['gp']['params'][i]
            rf_val = results_comparison['rf']['params'][i]
            diff = abs(gp_val - rf_val)
            print(f"{name:<12} {gp_val:<12.4f} {rf_val:<12.4f} {diff:<12.4f}")
        print("\nRENDIMIENTO:")
        print("-" * 50)
        print(f"{'Métrica':<20} {'GP':<15} {'RF':<15}")
        print("-" * 50)
        print(f"{'RMSE (MPa)':<20} {results_comparison['gp']['rmse']/1e6:<15.4f} {results_comparison['rf']['rmse']/1e6:<15.4f}")
        print(f"{'Error Promedio (%)':<20} {results_comparison['gp']['avg_error']:<15.2f} {results_comparison['rf']['avg_error']:<15.2f}")
        print(f"{'Dentro límites':<20} {str(results_comparison['gp']['in_bounds']):<15} {str(results_comparison['rf']['in_bounds']):<15}")
        print("\nERRORES POR REGIÓN:")
        print("-" * 50)
        regions = ['Proximal (%)', 'Medial (%)', 'Distal (%)']
        for i, region in enumerate(regions):
            gp_err = results_comparison['gp']['individual_errors'][i]
            rf_err = results_comparison['rf']['individual_errors'][i]
            print(f"{region:<20} {gp_err:<15.2f} {rf_err:<15.2f}")
        if results_comparison['gp']['avg_error'] < results_comparison['rf']['avg_error']:
            better_method = 'GP'
            better_results = results_comparison['gp']
        else:
            better_method = 'RF'
            better_results = results_comparison['rf']
        print(f"\nMÉTODO RECOMENDADO: {better_method}")
        print(f"Error promedio: {better_results['avg_error']:.2f}%")
        print(f"RMSE: {better_results['rmse']/1e6:.4f} MPa")
        return results_comparison, better_method

if __name__ == "__main__":
    optimizer = ProsthesisOptimizer('D:\GitHub\M_Sc\Extras\ProyectoMeli\Base_datos_protesis.csv')
    optimizer.preprocess_data()
    results_comparison, better_method = optimizer.compare_optimization_results()
    print(f"{' '*60}")
    print("PARÁMETROS FINALES RECOMENDADOS")
    print(f"{' '*60}")
    print(f"● Método recomendado: {better_method}")
    best_params = results_comparison[better_method.lower()]['params']
    param_names = ['Rproximal', 'Rdistal', 'incremento', 'Dst']
    print("● Parámetros óptimos:")
    for i, name in enumerate(param_names):
        print(f"  ◦ {name}: {best_params[i]:.4f}")
    print(f"● Error promedio: {results_comparison[better_method.lower()]['avg_error']:.2f}%")
