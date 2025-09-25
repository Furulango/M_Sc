import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import f_oneway
from tqdm import tqdm

warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

N_RUNS = 10
MAX_EVALUATIONS = 10000

TRUE_PARAMS = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
PARAM_NAMES = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
PARAM_BOUNDS = (TRUE_PARAMS * 0.5, TRUE_PARAMS * 1.5)

def induction_motor_model(t, x, params, vqs, vds):
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2 * np.pi * 60
    ws = we - wr
    
    lqs = Ls * iqs + Lm * iqr
    lds = Ls * ids + Lm * idr
    lqr = Lr * iqr + Lm * iqs
    ldr = Lr * idr + Lm * ids
    
    L_matrix = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    
    v_vector = np.array([
        vqs - rs * iqs - we * lds,
        vds - rs * ids + we * lqs,
        -rr * iqr - ws * ldr,
        -rr * idr + ws * lqr
    ])
    
    try:
        di_dt = np.linalg.solve(L_matrix, v_vector)
    except np.linalg.LinAlgError:
        di_dt = np.full(4, 1e6)

    Te = (3 * 4 / 4) * Lm * (iqs * idr - ids * iqr)
    dwr_dt = (Te - B * wr) / J
    
    return np.array([*di_dt, dwr_dt])

def simulate_motor(params, t_span=[0, 2], n_points=500):
    vqs, vds = 220 * np.sqrt(2) / np.sqrt(3), 0
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
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        
        Is_mag = np.sqrt(iqs**2 + ids**2)
        rpm = wr * 60 / (2 * np.pi) * (4 / 2)
        
        return t, {'current': Is_mag, 'rpm': rpm}

    except Exception:
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'current': np.full(n_points, 1e6), 'rpm': np.full(n_points, 1e6)}

class MotorObjective:
    def __init__(self, target_current, target_rpm):
        self.target_current = target_current
        self.target_rpm = target_rpm
        self.current_scale = np.max(target_current) if np.max(target_current) > 0 else 1.0
        self.rpm_scale = np.max(target_rpm) if np.max(target_rpm) > 0 else 1.0
        self.eval_count = 0

    def __call__(self, params):
        self.eval_count += 1
        
        if any(p <= 0 for p in params):
            return 1e12

        _, sim_outputs = simulate_motor(params, n_points=len(self.target_current))
        sim_current = sim_outputs['current']
        sim_rpm = sim_outputs['rpm']

        current_mse = np.mean(((self.target_current - sim_current) / self.current_scale)**2)
        rpm_mse = np.mean(((self.target_rpm - sim_rpm) / self.rpm_scale)**2)
        
        total_error = 0.7 * current_mse + 0.3 * rpm_mse
        
        return total_error if np.isfinite(total_error) else 1e12

class ParticleSwarmOptimizer:
    def __init__(self, objective_func, bounds, n_particles=50, max_iter=200):
        self.objective_func = objective_func
        self.lb, self.ub = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.n_dims = len(self.lb)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7

    def optimize(self):
        particles = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dims))
        velocities = np.zeros((self.n_particles, self.n_dims))
        
        pbest_pos = particles.copy()
        pbest_cost = np.array([self.objective_func(p) for p in particles])
        
        gbest_idx = np.argmin(pbest_cost)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]

        for _ in range(self.max_iter):
            r1, r2 = np.random.rand(2)
            
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_pos - particles) +
                          self.c2 * r2 * (gbest_pos - particles))
            
            particles += velocities
            particles = np.clip(particles, self.lb, self.ub)

            current_costs = np.array([self.objective_func(p) for p in particles])
            
            improved_mask = current_costs < pbest_cost
            pbest_pos[improved_mask] = particles[improved_mask]
            pbest_cost[improved_mask] = current_costs[improved_mask]
            
            if np.min(pbest_cost) < gbest_cost:
                gbest_idx = np.argmin(pbest_cost)
                gbest_pos = pbest_pos[gbest_idx].copy()
                gbest_cost = pbest_cost[gbest_idx]

        return gbest_cost, gbest_pos

class GreyWolfOptimizer:
    def __init__(self, objective_func, bounds, n_wolves=50, max_iter=200):
        self.objective_func = objective_func
        self.lb, self.ub = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_dims = len(self.lb)

    def optimize(self):
        alpha_pos = np.zeros(self.n_dims)
        alpha_score = float("inf")
        beta_pos = np.zeros(self.n_dims)
        beta_score = float("inf")
        delta_pos = np.zeros(self.n_dims)
        delta_score = float("inf")

        positions = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.n_dims))

        for it in range(self.max_iter):
            for i in range(self.n_wolves):
                fitness = self.objective_func(positions[i, :])
                if fitness < alpha_score:
                    alpha_score, alpha_pos = fitness, positions[i, :].copy()
                elif alpha_score < fitness < beta_score:
                    beta_score, beta_pos = fitness, positions[i, :].copy()
                elif beta_score < fitness < delta_score:
                    delta_score, delta_pos = fitness, positions[i, :].copy()
            
            a = 2 - it * (2 / self.max_iter)

            for i in range(self.n_wolves):
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos - positions[i, :])
                X1 = alpha_pos - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos - positions[i, :])
                X2 = beta_pos - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos - positions[i, :])
                X3 = delta_pos - A3 * D_delta

                positions[i, :] = (X1 + X2 + X3) / 3
            
            positions = np.clip(positions, self.lb, self.ub)

        return alpha_score, alpha_pos

def run_comparison_study():
    print("Generando datos de referencia del motor real...")
    _, true_outputs = simulate_motor(TRUE_PARAMS)
    target_current = true_outputs['current']
    target_rpm = true_outputs['rpm']
    
    noise_level = 0.01
    target_current += np.random.normal(0, np.std(target_current) * noise_level, len(target_current))
    target_rpm += np.random.normal(0, np.std(target_rpm) * noise_level, len(target_rpm))

    algorithms = {
        "PSO": ParticleSwarmOptimizer,
        "GWO": GreyWolfOptimizer
    }
    
    all_results = []

    for alg_name, AlgClass in algorithms.items():
        print(f"\n{'='*30}\nEjecutando {alg_name} ({N_RUNS} veces)\n{'='*30}")
        
        for i in tqdm(range(N_RUNS), desc=f"   {alg_name} Progress"):
            start_time = time.time()
            
            objective_function = MotorObjective(target_current, target_rpm)
            
            n_agents = 50
            max_iterations = MAX_EVALUATIONS // n_agents
            
            optimizer = AlgClass(objective_function, PARAM_BOUNDS, n_agents, max_iterations)
            
            best_cost, best_params = optimizer.optimize()
            
            elapsed_time = time.time() - start_time
            evaluations = objective_function.eval_count
            param_error = np.mean(np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)) * 100

            run_result = {
                'algorithm': alg_name,
                'run': i + 1,
                'cost': best_cost,
                'param_error_perc': param_error,
                'evaluations': evaluations,
                'time_s': elapsed_time
            }
            for p_name, p_val in zip(PARAM_NAMES, best_params):
                run_result[f'param_{p_name}'] = p_val
            
            all_results.append(run_result)

    results_df = pd.DataFrame(all_results)
    output_path = "results/pso_vs_gwo_comparison.csv"
    results_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nResultados guardados en: {output_path}")

    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO FINAL")
    print("="*80)
    
    summary = results_df.groupby('algorithm').agg({
        'param_error_perc': ['mean', 'std', 'min', 'max'],
        'time_s': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    print("\n" + "-"*80)
    print("Análisis de Varianza (ANOVA) para el error de parámetros:")
    
    pso_errors = results_df[results_df['algorithm'] == 'PSO']['param_error_perc']
    gwo_errors = results_df[results_df['algorithm'] == 'GWO']['param_error_perc']
    
    f_stat, p_value = f_oneway(pso_errors, gwo_errors)
    print(f"   P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("   -> Existe una diferencia estadísticamente significativa entre los algoritmos.")
    else:
        print("   -> No hay evidencia de una diferencia significativa entre los algoritmos.")
    print("-"*80)


if __name__ == "__main__":
    run_comparison_study()
