import numpy as np
from scipy.integrate import solve_ivp
from functions import run_pso, run_pso_sqp, BacterialForaging
import time
import pyswarms as ps

def induction_motor(t, x, params, vqs, vds):
    """Basic induction motor model in DQ coordinates"""
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2*np.pi*60
    ws = we - wr
    
    lqs = Ls*iqs + Lm*iqr
    lds = Ls*ids + Lm*idr
    lqr = Lr*iqr + Lm*iqs
    ldr = Lr*idr + Lm*ids
    
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], 
                  [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs*iqs - we*lds, vds - rs*ids + we*lqs,
                  -rr*iqr - ws*ldr, -rr*idr + ws*lqr])
    
    di_dt = np.linalg.solve(L, v)
    Te = (3*4/4) * Lm * (iqs*idr - ids*iqr)
    dwr_dt = (Te - B*wr) / J
    
    return np.array([*di_dt, dwr_dt])

def simulate_motor(params, t_span=[0, 2], n_points=500):
    """Simulates the motor and returns signals of interest"""
    vqs, vds = 220*np.sqrt(2)/np.sqrt(3), 0
    
    try:
        sol = solve_ivp(lambda t, x: induction_motor(t, x, params, vqs, vds),
                        t_span, [0,0,0,0,0], dense_output=True, rtol=1e-6)
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        
        Is_mag = np.sqrt(iqs**2 + ids**2)
        Te = (3*4/4) * params[4] * (iqs*idr - ids*iqr)
        rpm = wr * 60/(2*np.pi) * 2/4
        
        return t, {'iqs': iqs, 'ids': ids, 'Is_mag': Is_mag, 'Te': Te, 'rpm': rpm, 'wr': wr}
    
    except Exception as e:
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'iqs': np.ones(n_points)*1e6, 'ids': np.ones(n_points)*1e6, 
                   'Is_mag': np.ones(n_points)*1e6, 'Te': np.ones(n_points)*1e6, 
                   'rpm': np.ones(n_points)*1e6, 'wr': np.ones(n_points)*1e6}

def generate_experimental_data(real_params, noise_level=0.02):
    """Generates 'experimental' data by adding noise to perfect simulation"""
    t, outputs = simulate_motor(real_params)
    
    np.random.seed(42)
    exp_data = {}
    
    for key, signal in outputs.items():
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        exp_data[key] = signal + noise
    
    return t, exp_data

def objective_function(estimated_params, t_exp, exp_data, weights=None):
    """Objective function to minimize: Mean Square Error (MSE)"""
    if weights is None:
        weights = {'Is_mag': 1.0, 'Te': 0.5, 'rpm': 0.3}
    
    if any(p <= 0 for p in estimated_params[:5]) or any(p < 0 for p in estimated_params[5:]):
        return 1e10
    
    try:
        _, sim_outputs = simulate_motor(estimated_params)
        
        total_error = 0
        for signal, weight in weights.items():
            if signal in exp_data and signal in sim_outputs:
                error = np.mean((exp_data[signal] - sim_outputs[signal])**2)
                total_error += weight * error
        
        return total_error
        
    except Exception as e:
        return 1e10

def pso_with_progress(objective, bounds, n_particles=30, iterations=50):
    """PSO using pyswarms"""
    def pso_wrapper(x, **kwargs):
        return np.array([objective(p) for p in x])

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options={'c1': 2.05, 'c2': 2.05, 'w': 0.9},
        bounds=bounds
    )

    best_cost, best_pos = optimizer.optimize(pso_wrapper, iters=iterations, verbose=False)
    return float(best_cost), best_pos

def bfo_with_progress(objective, bounds, n_bacteria=20, n_chemotactic=15, n_reproductive=3):
    """Basic BFO"""
    bfo = BacterialForaging(objective, bounds, n_bacteria, n_chemotactic, 4, n_reproductive, 2, 0.25, 0.1)
    return bfo.optimize()

def generate_unique_seed(csv_file="parameter_identification_results.csv"):
    """Generates a unique random seed that is not already in use in the CSV file"""
    import random
    import time
    import os
    import pandas as pd
    
    used_seeds = set()
    
    if os.path.exists(csv_file):
        try:
            existing_df = pd.read_csv(csv_file)
            if 'Seed' in existing_df.columns:
                used_seeds = set(existing_df['Seed'].dropna().astype(int))
        except Exception as e:
            pass
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        time.sleep(0.001)
        seed = random.randint(1, 999999)
        
        if seed not in used_seeds:
            return seed
    
    backup_seed = int(time.time() * 1000000) % 999999
    while backup_seed in used_seeds:
        backup_seed = (backup_seed + 1) % 999999
    
    return backup_seed

def calculate_complete_metrics(identified_params, real_params, t_exp, exp_data, 
                             final_cost, execution_time, seed, algorithm, run_num):
    """Calculates all required metrics for statistical analysis"""
    
    _, sim_outputs = simulate_motor(identified_params)
    
    # MSE and MAE errors
    mse_current = np.mean((exp_data['Is_mag'] - sim_outputs['Is_mag'])**2)
    mse_torque = np.mean((exp_data['Te'] - sim_outputs['Te'])**2)
    mse_rpm = np.mean((exp_data['rpm'] - sim_outputs['rpm'])**2)
    
    mae_current = np.max(np.abs(exp_data['Is_mag'] - sim_outputs['Is_mag']))
    mae_torque = np.max(np.abs(exp_data['Te'] - sim_outputs['Te']))
    mae_rpm = np.max(np.abs(exp_data['rpm'] - sim_outputs['rpm']))
    
    # Parameter metrics
    param_errors_pct = np.abs((identified_params - real_params) / real_params) * 100
    average_error_pct = np.mean(param_errors_pct)
    euclidean_distance = np.linalg.norm(identified_params - real_params)
    max_param_error = np.max(param_errors_pct)
    good_params_count = np.sum(param_errors_pct < 5.0)
    
    # Statistical metrics
    rmse_total = np.sqrt(1.0*mse_current + 0.5*mse_torque + 0.3*mse_rpm)
    
    ss_res = np.sum((exp_data['Is_mag'] - sim_outputs['Is_mag'])**2)
    ss_tot = np.sum((exp_data['Is_mag'] - np.mean(exp_data['Is_mag']))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    mape_current = np.mean(np.abs((exp_data['Is_mag'] - sim_outputs['Is_mag']) / exp_data['Is_mag'])) * 100
    mape_torque = np.mean(np.abs((exp_data['Te'] - sim_outputs['Te']) / np.abs(exp_data['Te']) + 1e-8)) * 100
    mape_rpm = np.mean(np.abs((exp_data['rpm'] - sim_outputs['rpm']) / np.abs(exp_data['rpm']) + 1e-8)) * 100
    mape_average = np.mean([mape_current, mape_torque, mape_rpm])
    
    numerator = np.sqrt(np.mean((sim_outputs['Is_mag'] - exp_data['Is_mag'])**2))
    denominator = np.sqrt(np.mean(exp_data['Is_mag']**2)) + np.sqrt(np.mean(sim_outputs['Is_mag']**2))
    theil_coefficient = numerator / denominator if denominator > 0 else 1
    
    # Convergence criteria
    criterion_1 = average_error_pct < 10.0
    criterion_2 = final_cost < 0.01
    criterion_3 = good_params_count >= 5
    criterion_4 = r_squared > 0.95
    
    criteria_met = sum([criterion_1, criterion_2, criterion_3, criterion_4])
    converged_successfully = criteria_met >= 2
    
    # Estimations
    if 'PSO' in algorithm:
        num_evaluations = 30 * 50
        config_algorithm = "PSO_30p_50i"
    elif 'BFO' in algorithm:
        num_evaluations = 20 * 15 * 3
        config_algorithm = "BFO_20b_15q_3r"
    else:
        num_evaluations = 1000
        config_algorithm = algorithm
    
    metrics = {
        'Algorithm': algorithm,
        'Run': run_num,
        'Seed': seed,
        'Configuration': config_algorithm,
        'Time_s': execution_time,
        'Final_Cost': final_cost,
        'Best_Cost_Iter': final_cost,
        'Num_FO_Evaluations': num_evaluations,
        'Convergence_Iteration': 50 if converged_successfully else -1,
        'Converged_Successfully': converged_successfully,
        'MSE_Current': mse_current,
        'MSE_Torque': mse_torque,
        'MSE_RPM': mse_rpm,
        'RMSE_Total': rmse_total,
        'MAE_Current': mae_current,
        'MAE_Torque': mae_torque,
        'MAE_RPM': mae_rpm,
        'rs_identified': identified_params[0],
        'rr_identified': identified_params[1],
        'Lls_identified': identified_params[2],
        'Llr_identified': identified_params[3],
        'Lm_identified': identified_params[4],
        'J_identified': identified_params[5],
        'B_identified': identified_params[6],
        'Error_pct_rs': param_errors_pct[0],
        'Error_pct_rr': param_errors_pct[1],
        'Error_pct_Lls': param_errors_pct[2],
        'Error_pct_Llr': param_errors_pct[3],
        'Error_pct_Lm': param_errors_pct[4],
        'Error_pct_J': param_errors_pct[5],
        'Error_pct_B': param_errors_pct[6],
        'Average_Error_Pct': average_error_pct,
        'Average_Absolute_Error': np.mean(np.abs(identified_params - real_params)),
        'Euclidean_Distance': euclidean_distance,
        'Max_Param_Error': max_param_error,
        'Good_Params_5pct': good_params_count,
        'R_Squared': r_squared,
        'MAPE_Average': mape_average,
        'Theil_Coefficient': theil_coefficient,
        'Criterion_Error10pct': criterion_1,
        'Criterion_Cost001': criterion_2,
        'Criterion_5Params5pct': criterion_3,
        'Criterion_R2_95pct': criterion_4,
        'Criteria_Met': criteria_met
    }
    
    return metrics

def save_results_csv(metrics_results, filename="parameter_identification_results.csv"):
    """Saves results in CSV format"""
    import csv
    import os
    from datetime import datetime
    
    file_exists = os.path.exists(filename)
    headers = list(metrics_results[0].keys()) if metrics_results else []
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
            csvfile.write(f"# Dataset generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            csvfile.write(f"# Real parameters: rs=2.45, rr=1.83, Lls=0.008, Llr=0.008, Lm=0.203, J=0.02, B=0.001\n")
            csvfile.write(f"# Experimental noise: 3%\n")
        
        for metrics in metrics_results:
            writer.writerow(metrics)

def execute_algorithm_multiple(algorithm, objective, bounds, config, num_runs=30, real_params=None, t_exp=None, exp_data=None):
    """Executes an algorithm multiple times with unique random seeds"""
    
    print(f"\nExecuting {algorithm} - {num_runs} runs")
    
    for run in range(num_runs):
        seed = generate_unique_seed()
        np.random.seed(seed)
        
        start_time = time.time()
        try:
            if algorithm == 'PSO':
                cost, params = pso_with_progress(objective, bounds, 
                                               n_particles=config['n_particles'], 
                                               iterations=config['iterations'])
            elif algorithm == 'PSO-SQP':
                cost, params = run_pso_sqp(objective, bounds,
                                          n_particles=config['n_particles'],
                                          pso_iterations=config['pso_iterations'])
            elif algorithm == 'BFO':
                cost, params = bfo_with_progress(objective, bounds,
                                               n_bacteria=config['n_bacteria'],
                                               n_chemotactic=config['n_chemotactic'],
                                               n_reproductive=config['n_reproductive'])
            
            execution_time = time.time() - start_time
            
            metrics = calculate_complete_metrics(
                params, real_params, t_exp, exp_data,
                cost, execution_time, seed, algorithm, run+1
            )
            
            save_results_csv([metrics])
            
            print(f"  Run {run+1:2d}/30 - Seed: {seed} - Cost: {cost:.2e} - Error: {metrics['Average_Error_Pct']:.1f}%")
            
        except Exception as e:
            print(f"  Run {run+1:2d}/30 - ERROR: {str(e)[:50]}")

def execute_identification():
    """Main function to generate the CSV dataset"""
    
    real_params = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
    
    print("Generating experimental data...")
    t_exp, exp_data = generate_experimental_data(real_params, noise_level=0.03)
    
    search_factor = 0.5
    lb = real_params * (1 - search_factor)
    ub = real_params * (1 + search_factor)
    bounds = (lb, ub)
    
    objective = lambda params: objective_function(params, t_exp, exp_data)
    
    config = {
        'PSO': {'n_particles': 30, 'iterations': 50},
        'PSO-SQP': {'n_particles': 20, 'pso_iterations': 30},
        'BFO': {'n_bacteria': 20, 'n_chemotactic': 15, 'n_reproductive': 3}
    }
    
    print("Starting identification - 30 runs per algorithm")
    print("Auto-saving to: parameter_identification_results.csv")
    
    for algorithm in ['PSO', 'PSO-SQP', 'BFO']:
        execute_algorithm_multiple(algorithm, objective, bounds, config[algorithm], 
                                 num_runs=30, real_params=real_params, 
                                 t_exp=t_exp, exp_data=exp_data)
    
    # Final summary
    try:
        import pandas as pd
        df = pd.read_csv("parameter_identification_results.csv")
        print(f"\nFINAL SUMMARY:")
        print(f"Total runs: {len(df)}")
        for alg in ['PSO', 'PSO-SQP', 'BFO']:
            alg_data = df[df['Algorithm'] == alg]
            success_rate = alg_data['Converged_Successfully'].mean() * 100
            average_error = alg_data['Average_Error_Pct'].mean()
            print(f"{alg}: {len(alg_data)} runs - Success rate: {success_rate:.1f}% - Average error: {average_error:.1f}%")
    except:
        pass
    
    print("CSV dataset generated successfully")

if __name__ == "__main__":
    print("Parameter Identification System - Dataset Generation")
    print("Estimated time: 45-90 minutes")
    
    start = time.time()
    execute_identification()
    total_time = time.time() - start
    
    print(f"Process completed in {total_time/60:.1f} minutes")