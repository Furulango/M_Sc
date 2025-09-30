import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('results', exist_ok=True)

# reproducibilidad (opcional)
np.random.seed(0)

N_RUNS = 3  # Número de ejecuciones por algoritmo
TRUE_PARAMS = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
PARAM_NAMES = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
PARAM_BOUNDS = (TRUE_PARAMS * 0.8, TRUE_PARAMS * 1.2)

# -----------------------------------------------------------------------------
# Motor Model
# -----------------------------------------------------------------------------

def induction_motor_model(t, x, params, vqs, vds):
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2 * np.pi * 60
    # En el marco de referencia del estator, la velocidad síncrona es 'we'.
    # La velocidad de deslizamiento 'ws' es la diferencia entre la síncrona y la del rotor.
    ws = we - wr

    # Flujos concatenados
    lqs = Ls * iqs + Lm * iqr
    lds = Ls * ids + Lm * idr
    lqr = Lr * iqr + Lm * iqs
    ldr = Lr * idr + Lm * ids

    # Matriz de inductancias
    L_matrix = np.array([[Ls, 0, Lm, 0],
                         [0, Ls, 0, Lm],
                         [Lm, 0, Lr, 0],
                         [0, Lm, 0, Lr]])
    
    # Vector de tensiones y términos resistivos/de rotación
    v_vector = np.array([
        vqs - rs * iqs,
        vds - rs * ids,
        -rr * iqr - ws * ldr, # Término correcto es ws*ldr
        -rr * idr + ws * lqr  # Término correcto es ws*lqr
    ])

    # Resolver para las derivadas de las corrientes
    di_dt = np.linalg.solve(L_matrix, v_vector)
    
    Te = (3/2) * (4/2) * Lm * (iqs * idr - ids * iqr)
    dwr_dt = (Te - B * wr) / J

    return np.array([*di_dt, dwr_dt])


def simulate_motor(params, t_span=[0, 2], n_points=500):
    """
    Simula el motor y devuelve (t_eval, Is_mag, rpm, Te)
    """
    vqs = 220 * np.sqrt(2) / np.sqrt(3)
    vds = 0
    initial_state = [0, 0, 0, 0, 0]

    try:
        sol = solve_ivp(
            fun=lambda t, x: induction_motor_model(t, x, params, vqs, vds),
            t_span=t_span,
            y0=initial_state,
            method='RK45',
            dense_output=True,
            rtol=1e-6,
            atol=1e-8
        )

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t_eval)
        Lm = params[4]

        Is_mag = np.sqrt(iqs**2 + ids**2)
        rpm = wr * 60 / (2 * np.pi)
        Te = (3/2) * (4/2) * Lm * (iqs * idr - ids * iqr)

        return t_eval, Is_mag, rpm, Te
    except Exception as e:
        n_points = n_points if isinstance(n_points, int) else 500
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        return t_eval, np.ones(n_points) * 1e6, np.zeros(n_points), np.zeros(n_points)

# -----------------------------------------------------------------------------
# Objective Function
# -----------------------------------------------------------------------------

class FullObjective:
    def __init__(self, target_current, target_rpm, target_torque):
        self.target_current = target_current
        self.target_rpm = target_rpm
        self.target_torque = target_torque
        self.eval_count = 0

    def __call__(self, params):
        self.eval_count += 1
        if any(p <= 0 for p in params):
            return 1e10

        _, sim_current, sim_rpm, sim_torque = simulate_motor(params)

        current_error = np.mean((self.target_current - sim_current)**2)
        rpm_error = np.mean((self.target_rpm - sim_rpm)**2)
        torque_error = np.mean((self.target_torque - sim_torque)**2)

        return current_error + rpm_error * 0.001 + torque_error * 0.01

# -----------------------------------------------------------------------------
# Optimization Algorithms
# -----------------------------------------------------------------------------

class SimplePSO:
    def __init__(self, objective_func, bounds, n_particles=40, max_iter=50, c1=1.496, c2=1.496, w_max=0.9, w_min=0.4):
        self.obj = objective_func
        self.lb, self.ub = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.n_dims = len(self.lb)
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min

    def optimize(self):
        X = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dims))
        V = np.zeros((self.n_particles, self.n_dims))
        pbest = X.copy()
        pbest_cost = np.array([self.obj(x) for x in X])
        gbest_idx = np.argmin(pbest_cost)
        gbest = pbest[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]
        
        cost_history = [gbest_cost]

        print(f"      Iteración inicial - mejor costo: {gbest_cost:.6f}")
        vmax = (self.ub - self.lb) * 0.2

        for it in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * (it / max(1, (self.max_iter - 1)))
            r1 = np.random.rand(self.n_particles, self.n_dims)
            r2 = np.random.rand(self.n_particles, self.n_dims)

            V = w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            V = np.clip(V, -vmax, vmax)
            X = np.clip(X + V, self.lb, self.ub)

            for i in range(self.n_particles):
                cost = self.obj(X[i])
                if cost < pbest_cost[i]:
                    pbest[i] = X[i].copy()
                    pbest_cost[i] = cost
                    if cost < gbest_cost:
                        gbest = X[i].copy()
                        gbest_cost = cost
            
            cost_history.append(gbest_cost)
            if it % 10 == 0:
                print(f"      Iteración {it}/{self.max_iter} - mejor costo: {gbest_cost:.6f}")

        return gbest_cost, gbest, cost_history


class SimpleGWO:
    def __init__(self, objective_func, bounds, n_wolves=40, max_iter=50):
        self.obj = objective_func
        self.lb, self.ub = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_dims = len(self.lb)

    def optimize(self):
        X = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.n_dims))
        fitness = np.array([self.obj(x) for x in X])
        
        sorted_idx = np.argsort(fitness)
        alpha_pos, beta_pos, delta_pos = X[sorted_idx[0]].copy(), X[sorted_idx[1]].copy(), X[sorted_idx[2]].copy()
        alpha_score = fitness[sorted_idx[0]]
        
        cost_history = [alpha_score]

        print(f"      Iteración inicial - mejor costo: {alpha_score:.6f}")

        for it in range(self.max_iter):
            a = 2 - it * (2 / max(1, self.max_iter))
            for i in range(self.n_wolves):
                r1, r2 = np.random.rand(2)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X[i])

                r1, r2 = np.random.rand(2)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - X[i])

                r1, r2 = np.random.rand(2)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X[i])

                X[i] = np.clip((X1 + X2 + X3) / 3, self.lb, self.ub)

            for i in range(self.n_wolves):
                 fitness[i] = self.obj(X[i])

            sorted_idx = np.argsort(fitness)
            current_alpha_score = fitness[sorted_idx[0]]
            
            if current_alpha_score < alpha_score:
                alpha_score = current_alpha_score
                alpha_pos = X[sorted_idx[0]].copy()
                beta_pos = X[sorted_idx[1]].copy()
                delta_pos = X[sorted_idx[2]].copy()

            cost_history.append(alpha_score)

            if it % 10 == 0:
                print(f"      Iteración {it}/{self.max_iter} - mejor costo: {alpha_score:.6f}")

        return alpha_score, alpha_pos, cost_history

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------

def plot_best_result(best_params, algorithm_name, target_t, target_current, target_rpm, target_torque):
    t_sim, sim_current, sim_rpm, sim_torque = simulate_motor(best_params, t_span=[target_t[0], target_t[-1]], n_points=len(target_t))

    plt.figure(figsize=(21, 6))
    plt.subplot(1, 3, 1)
    plt.plot(target_t, target_current, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_current, '--', linewidth=2.5, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Corriente ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Corriente (A)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(target_t, target_rpm, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_rpm, '--', linewidth=2.5, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Velocidad ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (RPM)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(target_t, target_torque, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_torque, '--', linewidth=2.5, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Torque ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Mejor Resultado de Optimización con {algorithm_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = f'results/best_result_{algorithm_name}.svg'
    plt.savefig(plot_filename, format='svg', dpi=300)
    print(f"    -> Gráfica de mejor resultado guardada en: {plot_filename}")
    plt.close()

def plot_convergence(results_df):
    plt.figure(figsize=(10, 7))
    for alg_name in results_df['algorithm'].unique():
        alg_data = results_df[results_df['algorithm'] == alg_name]
        best_run_idx = alg_data['cost'].idxmin()
        best_run_history = alg_data.loc[best_run_idx, 'cost_history']
        plt.plot(best_run_history, label=alg_name, linewidth=2.5)

    plt.title('Curvas de Convergencia (Mejor Corrida)')
    plt.xlabel('Iteración')
    plt.ylabel('Valor de la Función de Coste (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plot_filename = 'results/convergence_plot.svg'
    plt.savefig(plot_filename, format='svg', dpi=300)
    print(f"    -> Gráfica de convergencia guardada en: {plot_filename}")
    plt.close()

def plot_error_distribution(results_df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='algorithm', y='param_error_%', data=results_df)
    plt.title(f'Distribución del Error de Parámetros ({N_RUNS} Corridas)')
    plt.ylabel('Error Promedio de Parámetros (%)')
    plt.xlabel('Algoritmo')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plot_filename = 'results/error_distribution_boxplot.svg'
    plt.savefig(plot_filename, format='svg', dpi=300)
    print(f"    -> Gráfica de distribución de error guardada en: {plot_filename}")
    plt.close()

def plot_parameter_errors(results_df):
    n_params = len(PARAM_NAMES)
    index = np.arange(n_params)
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    algorithms = results_df['algorithm'].unique()
    for i, alg_name in enumerate(algorithms):
        alg_data = results_df[results_df['algorithm'] == alg_name]
        best_run_idx = alg_data['param_error_%'].idxmin()
        best_params = alg_data.loc[best_run_idx, 'best_params']
        errors = 100 * np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)
        ax.bar(index + i * bar_width - bar_width/2, errors, bar_width, label=alg_name)

    ax.set_ylabel('Error de Estimación (%)')
    ax.set_title('Error por Parámetro Individual (Mejor Corrida)')
    ax.set_xticks(index)
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plot_filename = 'results/parameter_error_bars.svg'
    plt.savefig(plot_filename, format='svg', dpi=300)
    print(f"    -> Gráfica de error por parámetro guardada en: {plot_filename}")
    plt.close()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    print("="*60)
    print("      ANÁLISIS COMPARATIVO DE ALGORITMOS DE OPTIMIZACIÓN")
    print("="*60)

    print("\n[1] Generando datos de referencia...")
    target_t, target_current, target_rpm, target_torque = simulate_motor(TRUE_PARAMS, t_span=[0, 2], n_points=500)

    noise_percentage = 0.15  # Nivel de ruido en porcentaje
    noise_c = np.random.normal(0, np.max(target_current) * noise_percentage, len(target_current))
    noise_r = np.random.normal(0, np.max(target_rpm) * noise_percentage, len(target_rpm))
    noise_t = np.random.normal(0, np.max(target_torque) * noise_percentage, len(target_torque))

    target_current_noisy = target_current + noise_c
    target_rpm_noisy = target_rpm + noise_r
    target_torque_noisy = target_torque + noise_t
    print(f"    Datos generados y ruido del {noise_percentage * 100}% aplicado.")

    results = []
    algorithms = {'PSO': SimplePSO, 'GWO': SimpleGWO}

    for alg_name, optimizer_class in algorithms.items():
        print(f"\n[2] Ejecutando optimización con {alg_name}...")
        for run in range(N_RUNS):
            print(f"    Run {run+1}/{N_RUNS}:")
            start = time.time()
            obj = FullObjective(target_current_noisy, target_rpm_noisy, target_torque_noisy)
            optimizer = optimizer_class(obj, PARAM_BOUNDS, max_iter=100) # Aumentado a 100 iteraciones
            best_cost, best_params, cost_history = optimizer.optimize()
            elapsed = time.time() - start

            param_error = np.mean(np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)) * 100
            
            results.append({
                'algorithm': alg_name,
                'run': run + 1,
                'cost': float(best_cost),
                'param_error_%': float(param_error),
                'time_s': float(elapsed),
                'best_params': best_params,
                'cost_history': cost_history
            })
            print(f"    Completado en {elapsed:.2f}s - Error final de parámetros: {param_error:.2f}%")
    
    df = pd.DataFrame(results)
    df.to_csv('results/full_results_with_history.csv', index=False)
    
    print("\n[3] Resumen Estadístico de Resultados")
    print("="*60)
    summary = df.groupby('algorithm').agg({
        'param_error_%': ['mean', 'min', 'std'],
        'time_s': ['mean', 'std']
    }).reset_index()

    summary.columns = ['Algoritmo', 'Error Promedio (%)', 'Mejor Error (%)', 'Desv. Est. Error', 'Tiempo Promedio (s)', 'Desv. Est. Tiempo']
    print(summary.to_string(index=False))
    print("="*60)

    print("\n[4] Generando gráficas de alta calidad...")
    # Graficar mejor resultado de cada algoritmo
    for alg_name in algorithms.keys():
        alg_data = df[df['algorithm'] == alg_name]
        best_run_idx = alg_data['param_error_%'].idxmin()
        best_run_params = alg_data.loc[best_run_idx, 'best_params']
        plot_best_result(best_run_params, alg_name, target_t, target_current_noisy, target_rpm_noisy, target_torque_noisy)

    # Generar gráficas de análisis adicionales
    plot_convergence(df)
    plot_error_distribution(df)
    plot_parameter_errors(df)
    
    print("\nAnálisis completado. Todos los resultados y gráficas se encuentran en la carpeta 'results/'.")

if __name__ == "__main__":
    main()
