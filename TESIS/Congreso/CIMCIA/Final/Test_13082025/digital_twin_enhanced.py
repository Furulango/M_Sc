# academic_digital_twin_comparison_enhanced.py
# COMPARATIVE STUDY: Bio-Inspired Algorithms for Digital Twin Parameter Adaptation
# Enhanced with CSV Export and Motor Performance Comparison Plots
# For Mechatronics, Control & AI Conference Submission
# 
# Authors: [Your Name]
# Institution: [Your Institution]

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create directories for results
os.makedirs('results', exist_ok=True)
os.makedirs('results/csv', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# ===============================================================================
# MOTOR SIMULATION FRAMEWORK
# ===============================================================================

def induction_motor(t, x, params, vqs, vds):
    """Induction motor model in DQ coordinates"""
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
    """Motor simulation with robust error handling"""
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
    
    except Exception:
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'iqs': np.ones(n_points)*1e6, 'ids': np.ones(n_points)*1e6, 
                   'Is_mag': np.ones(n_points)*1e6, 'Te': np.ones(n_points)*1e6, 
                   'rpm': np.ones(n_points)*1e6, 'wr': np.ones(n_points)*1e6}

# ===============================================================================
# BIO-INSPIRED OPTIMIZATION ALGORITHMS WITH PARAMETER TRACKING
# ===============================================================================

class StandardPSO:
    """Standard Particle Swarm Optimization using PySwarms with parameter tracking"""
    
    def __init__(self, objective_func, bounds, n_particles=30, max_iter=50):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.n_dims = len(bounds[0])
        
        # Progress tracking
        self.cost_history = []
        self.convergence_iteration = -1
        self.best_cost = float('inf')
        self.best_params = None  # Store best parameters
        
        # PySwarms options
        self.options = {'c1': 2.05, 'c2': 2.05, 'w': 0.9}
        
    def pso_objective_wrapper(self, x):
        """Wrapper for PySwarms objective function"""
        costs = np.array([self.objective_func(particle) for particle in x])
        
        # Track best cost and parameters
        current_best_idx = np.argmin(costs)
        current_best = costs[current_best_idx]
        
        if current_best < self.best_cost:
            self.best_cost = current_best
            self.best_params = x[current_best_idx].copy()  # Store parameters
            if self.convergence_iteration == -1 and current_best < 1e-6:
                self.convergence_iteration = len(self.cost_history)
        
        self.cost_history.append(current_best)
        
        return costs
    
    def optimize(self):
        """Execute PSO optimization with parameter tracking"""
        start_time = time.time()
        
        print(f"    PSO: Starting optimization...")
        
        # Initialize optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.n_dims,
            options=self.options,
            bounds=self.bounds
        )
        
        # Perform optimization
        best_cost, best_pos = optimizer.optimize(
            self.pso_objective_wrapper, 
            iters=self.max_iter,
            verbose=False
        )
        
        optimization_time = time.time() - start_time
        
        print(f"    PSO: Completed - Best cost: {best_cost:.2e}, Time: {optimization_time:.2f}s")
        
        return best_cost, best_pos, optimization_time

class BacterialForagingOptimization:
    """Bacterial Foraging Optimization with parameter tracking"""
    
    def __init__(self, objective_func, bounds, n_bacteria=30, n_chemotactic=50, 
                 n_swim=4, n_reproductive=4, n_elimination=2, p_eliminate=0.25, step_size=0.1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.S = n_bacteria
        self.Nc = n_chemotactic
        self.Ns = n_swim
        self.Nre = n_reproductive
        self.Ned = n_elimination
        self.Ped = p_eliminate
        self.Ci = step_size
        self.n_dims = len(bounds[0])
        self.lb, self.ub = bounds

        self.bacteria = np.random.uniform(self.lb, self.ub, (self.S, self.n_dims))
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.health = np.zeros(self.S)
        
        self.best_pos = self.bacteria[np.argmin(self.costs)]
        self.best_cost = np.min(self.costs)
        self.best_params = self.best_pos.copy()  # Store best parameters
        
        # Progress tracking
        self.cost_history = [self.best_cost]
        self.convergence_iteration = -1

    def optimize(self):
        """Execute BFO optimization with parameter tracking"""
        start_time = time.time()
        iteration_count = 0
        
        print(f"    BFO: Starting optimization...")
        
        for l in range(self.Ned):
            for k in range(self.Nre):
                for j in range(self.Nc):
                    iteration_count += 1
                    self._update_best()
                    
                    if self.convergence_iteration == -1 and self.best_cost < 1e-6:
                        self.convergence_iteration = iteration_count
                    
                    last_costs = np.copy(self.costs)
                    directions = self._tumble()
                    
                    # Swimming phase
                    for m in range(self.Ns):
                        new_pos = self.bacteria + self.Ci * directions
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_costs = np.array([self.objective_func(p) for p in new_pos])
                        
                        improved_mask = new_costs < self.costs
                        self.bacteria[improved_mask] = new_pos[improved_mask]
                        self.costs[improved_mask] = new_costs[improved_mask]
                        self.health += last_costs - self.costs
                        
                        if not np.any(improved_mask):
                            break

                    self.cost_history.append(self.best_cost)
                    
                    # Progress update every 10 iterations
                    if iteration_count % 10 == 0:
                        progress = (iteration_count / (self.Ned * self.Nre * self.Nc)) * 100
                        print(f"    BFO: {progress:.0f}% - Best cost: {self.best_cost:.2e}")

                self._reproduce()
            self._eliminate_disperse()
            
        self._update_best()
        optimization_time = time.time() - start_time
        
        print(f"    BFO: Completed - Best cost: {self.best_cost:.2e}, Time: {optimization_time:.2f}s")
        
        return self.best_cost, self.best_pos, optimization_time

    def _update_best(self):
        min_cost_idx = np.argmin(self.costs)
        if self.costs[min_cost_idx] < self.best_cost:
            self.best_cost = self.costs[min_cost_idx]
            self.best_pos = self.bacteria[min_cost_idx]
            self.best_params = self.best_pos.copy()  # Update best parameters

    def _tumble(self):
        direction = np.random.uniform(-1, 1, (self.S, self.n_dims))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        return direction / norm

    def _reproduce(self):
        sorted_indices = np.argsort(self.health)
        n_survive = self.S // 2
        
        survivors_pos = self.bacteria[sorted_indices[:n_survive]]
        self.bacteria = np.concatenate([survivors_pos, survivors_pos])
        
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.health = np.zeros(self.S)

    def _eliminate_disperse(self):
        for i in range(self.S):
            if np.random.rand() < self.Ped:
                self.bacteria[i] = np.random.uniform(self.lb, self.ub, self.n_dims)
                self.costs[i] = self.objective_func(self.bacteria[i])

class ChaoticPSODSO:
    """Chaotic PSO with Dynamic Self-Optimization with parameter tracking"""
    
    def __init__(self, objective_func, bounds, n_particles=30, max_iter=50):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.n_dims = len(bounds[0])
        
        # Dynamic parameters
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_init = 2.5
        self.c2_init = 0.5
        
        # Chaotic maps parameters
        self.chaos_param = 4.0
        self.chaos_values = np.random.rand(n_particles)
        
        # Initialize particles
        self.particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.n_dims))
        self.velocities = np.zeros((n_particles, self.n_dims))
        self.pbest = self.particles.copy()
        self.pbest_costs = np.array([objective_func(p) for p in self.particles])
        self.gbest = self.pbest[np.argmin(self.pbest_costs)]
        self.gbest_cost = np.min(self.pbest_costs)
        self.best_params = self.gbest.copy()  # Store best parameters
        
        # Progress tracking
        self.cost_history = [self.gbest_cost]
        self.convergence_iteration = -1
        
    def optimize(self):
        """Execute Chaotic PSO-DSO optimization with parameter tracking"""
        start_time = time.time()
        
        print(f"    Chaotic PSO-DSO: Starting optimization...")
        
        for iteration in range(self.max_iter):
            # Dynamic parameter adjustment
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
            c1 = (self.c1_init - 2.0) * iteration / self.max_iter + 2.0
            c2 = (2.0 - self.c2_init) * iteration / self.max_iter + self.c2_init
            
            for i in range(self.n_particles):
                # Update chaotic value
                self.chaos_values[i] = self.chaos_param * self.chaos_values[i] * (1 - self.chaos_values[i])
                
                # Chaotic velocity update
                r1 = self.chaos_values[i]
                r2 = np.random.rand()
                
                # Adaptive velocity update with chaos
                chaos_factor = 0.1 * self.chaos_values[i]
                
                self.velocities[i] = (w * self.velocities[i] + 
                                    c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                    c2 * r2 * (self.gbest - self.particles[i]) +
                                    chaos_factor * (np.random.rand(self.n_dims) - 0.5))
                
                # Update position
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.bounds[0], self.bounds[1])
                
                # Evaluate fitness
                cost = self.objective_func(self.particles[i])
                
                # Update personal best
                if cost < self.pbest_costs[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_costs[i] = cost
                    
                    # Update global best
                    if cost < self.gbest_cost:
                        self.gbest = self.particles[i].copy()
                        self.gbest_cost = cost
                        self.best_params = self.gbest.copy()  # Update best parameters
                        if self.convergence_iteration == -1 and cost < 1e-6:
                            self.convergence_iteration = iteration
            
            self.cost_history.append(self.gbest_cost)
            
            # Progress update every 10 iterations
            if (iteration + 1) % 10 == 0:
                progress = ((iteration + 1) / self.max_iter) * 100
                print(f"    Chaotic PSO-DSO: {progress:.0f}% - Best cost: {self.gbest_cost:.2e}")
        
        optimization_time = time.time() - start_time
        
        print(f"    Chaotic PSO-DSO: Completed - Best cost: {self.gbest_cost:.2e}, Time: {optimization_time:.2f}s")
        
        return self.gbest_cost, self.gbest, optimization_time

# ===============================================================================
# ENHANCED DIGITAL TWIN FRAMEWORK WITH CSV EXPORT AND MOTOR COMPARISON
# ===============================================================================

class DigitalTwinComparator:
    """Digital Twin framework with enhanced results export and motor comparison"""
    
    def __init__(self, ideal_params):
        self.ideal_params = np.array(ideal_params)
        
        # Parameter names for CSV export
        self.param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
        
        # Temperature compensation coefficients
        self.temp_coeffs = np.array([0.004, 0.004, 0.001, 0.001, 0.0005, 0, 0])
        self.reference_temp = 20.0
        
        # Algorithm instances
        self.algorithms = {
            'PSO': StandardPSO,
            'BFO': BacterialForagingOptimization,
            'Chaotic PSO-DSO': ChaoticPSODSO
        }
        
        # Results storage
        self.results = {}
        self.convergence_curves = {}
        self.detailed_results = {}  # Store detailed parameters for each run
        self.scenario_data_storage = {}  # Store scenario data for motor comparison
        
    def compensate_temperature(self, params, temperature):
        """Apply temperature compensation"""
        temp_diff = temperature - self.reference_temp
        compensated = params.copy()
        
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        
        return compensated
    
    def generate_nonideal_scenario(self, scenario_name, temperature, degradation_factor, noise_level):
        """Generate non-ideal motor scenario for testing"""
        
        # Create non-ideal parameters
        nonideal_params = self.ideal_params.copy()
        
        # Apply temperature effects
        nonideal_params = self.compensate_temperature(nonideal_params, temperature)
        
        # Apply random degradation
        np.random.seed(hash(scenario_name) % 2**32)
        degradation = np.random.normal(1.0, degradation_factor, len(nonideal_params))
        degradation = np.clip(degradation, 0.7, 1.3)
        nonideal_params *= degradation
        
        # Simulate motor
        t, outputs = simulate_motor(nonideal_params, t_span=[0, 1.5], n_points=300)
        
        # Add measurement noise to current signal
        current_clean = outputs['Is_mag']
        noise = np.random.normal(0, noise_level * np.std(current_clean), len(current_clean))
        current_measured = current_clean + noise
        
        return {
            'true_params': nonideal_params,
            'measured_current': current_measured,
            'temperature': temperature,
            'time': t,
            'all_outputs': outputs
        }
    
    def create_objective_function(self, measured_current, temperature):
        """Create objective function for parameter identification"""
        
        def objective(candidate_params):
            try:
                # Apply temperature compensation
                temp_compensated = self.compensate_temperature(candidate_params, temperature)
                
                # Simulate with candidate parameters
                _, sim_outputs = simulate_motor(temp_compensated, 
                                              t_span=[0, len(measured_current)*0.005],
                                              n_points=len(measured_current))
                
                sim_current = sim_outputs['Is_mag']
                
                # MSE between measured and simulated current
                mse = np.mean((measured_current - sim_current)**2)
                
                return mse
                
            except Exception:
                return 1e10
        
        return objective
    
    def run_comparative_study(self, n_runs=10):
        """Execute comprehensive comparative study with detailed parameter tracking"""
        
        print("="*80)
        print("COMPARATIVE STUDY: Bio-Inspired Algorithms for Digital Twin")
        print("Enhanced with CSV Export and Motor Performance Comparison")
        print("="*80)
        
        # Test scenarios
        scenarios = [
            {'name': 'Normal_Operation', 'temp': 40, 'degradation': 0.05, 'noise': 0.02},
            {'name': 'High_Temperature', 'temp': 70, 'degradation': 0.10, 'noise': 0.03},
            {'name': 'Severe_Conditions', 'temp': 85, 'degradation': 0.15, 'noise': 0.04}
        ]
        
        # Initialize results storage
        for algorithm in self.algorithms.keys():
            self.results[algorithm] = {
                'costs': [],
                'errors': [],
                'times': [],
                'convergence_iterations': [],
                'robustness_scores': []
            }
            self.convergence_curves[algorithm] = []
            self.detailed_results[algorithm] = []
        
        # Search bounds
        search_factor = 0.4
        bounds = (self.ideal_params * (1 - search_factor), 
                 self.ideal_params * (1 + search_factor))
        
        # Execute comparative study
        for scenario in scenarios:
            print(f"\n" + "="*60)
            print(f"TESTING SCENARIO: {scenario['name']}")
            print(f"Temperature: {scenario['temp']}°C, Noise: {scenario['noise']*100}%")
            print("="*60)
            
            # Generate scenario data
            scenario_data = self.generate_nonideal_scenario(
                scenario['name'], scenario['temp'], 
                scenario['degradation'], scenario['noise']
            )
            
            # Store scenario data for motor comparison
            self.scenario_data_storage[scenario['name']] = scenario_data
            
            # Create objective function
            objective = self.create_objective_function(
                scenario_data['measured_current'], 
                scenario_data['temperature']
            )
            
            # Test each algorithm
            for alg_name, AlgorithmClass in self.algorithms.items():
                print(f"\n  Algorithm: {alg_name}")
                print(f"  " + "-" * 40)
                
                scenario_costs = []
                scenario_errors = []
                scenario_times = []
                scenario_convergence = []
                scenario_curves = []
                
                # Multiple runs for statistical significance
                for run in range(n_runs):
                    print(f"\n  Run {run+1}/{n_runs}:")
                    
                    np.random.seed(run * 100 + hash(alg_name + scenario['name']) % 1000)
                    
                    # Initialize algorithm
                    if alg_name == 'BFO':
                        algorithm = AlgorithmClass(objective, bounds, n_bacteria=30, 
                                                 n_chemotactic=25, n_reproductive=3)
                    else:
                        algorithm = AlgorithmClass(objective, bounds, n_particles=30, max_iter=30)
                    
                    # Execute optimization
                    cost, params, opt_time = algorithm.optimize()
                    
                    # Calculate parameter error
                    param_error = np.mean(np.abs((params - scenario_data['true_params']) / 
                                                scenario_data['true_params']) * 100)
                    
                    # Store detailed results
                    detailed_run = {
                        'scenario': scenario['name'],
                        'run': run + 1,
                        'cost': cost,
                        'error': param_error,
                        'time': opt_time,
                        'identified_params': params.tolist(),
                        'true_params': scenario_data['true_params'].tolist()
                    }
                    
                    # Add parameter names
                    for i, param_name in enumerate(self.param_names):
                        detailed_run[f'identified_{param_name}'] = params[i]
                        detailed_run[f'true_{param_name}'] = scenario_data['true_params'][i]
                    
                    self.detailed_results[alg_name].append(detailed_run)
                    
                    # Store results
                    scenario_costs.append(cost)
                    scenario_errors.append(param_error)
                    scenario_times.append(opt_time)
                    scenario_curves.append(algorithm.cost_history)
                    
                    if hasattr(algorithm, 'convergence_iteration'):
                        scenario_convergence.append(algorithm.convergence_iteration)
                    else:
                        scenario_convergence.append(-1)
                    
                    print(f"    Result: Error={param_error:.2f}%, Cost={cost:.2e}, Time={opt_time:.2f}s")
                
                # Calculate robustness
                robustness = 1 / (np.std(scenario_errors) / np.mean(scenario_errors) + 1e-8)
                
                # Store aggregated results
                self.results[alg_name]['costs'].extend(scenario_costs)
                self.results[alg_name]['errors'].extend(scenario_errors)
                self.results[alg_name]['times'].extend(scenario_times)
                self.results[alg_name]['convergence_iterations'].extend(scenario_convergence)
                self.results[alg_name]['robustness_scores'].append(robustness)
                self.convergence_curves[alg_name].extend(scenario_curves)
                
                # Print scenario summary
                print(f"\n  SCENARIO SUMMARY for {alg_name}:")
                print(f"    Mean Error: {np.mean(scenario_errors):.2f}% ± {np.std(scenario_errors):.2f}%")
                print(f"    Mean Time: {np.mean(scenario_times):.3f}s")
                print(f"    Robustness: {robustness:.3f}")
        
        # Export results to CSV
        self.export_to_csv()
        
        return self.results
    
    def export_to_csv(self):
        """Export detailed results to CSV files"""
        print("\n" + "="*80)
        print("EXPORTING RESULTS TO CSV")
        print("="*80)
        
        for alg_name, results in self.detailed_results.items():
            if results:
                # Create DataFrame
                df = pd.DataFrame(results)
                
                # Reorder columns for better readability
                base_cols = ['scenario', 'run', 'cost', 'error', 'time']
                param_cols = [f'identified_{name}' for name in self.param_names]
                true_cols = [f'true_{name}' for name in self.param_names]
                
                ordered_cols = base_cols + param_cols + true_cols
                df = df[ordered_cols]
                
                # Save to CSV
                filename = f"results/csv/{alg_name.replace(' ', '_')}_results.csv"
                df.to_csv(filename, index=False, float_format='%.6f')
                print(f"  ✓ Saved {alg_name} results to {filename}")
                
                # Print summary statistics
                print(f"    - Total runs: {len(df)}")
                print(f"    - Mean error: {df['error'].mean():.2f}%")
                print(f"    - Mean time: {df['time'].mean():.3f}s")
    
    def plot_motor_comparison(self):
        """Create motor performance comparison plots for each scenario"""
        print("\n" + "="*80)
        print("GENERATING MOTOR PERFORMANCE COMPARISON PLOTS")
        print("="*80)
        
        scenarios = ['Normal_Operation', 'High_Temperature', 'Severe_Conditions']
        
        for scenario_name in scenarios:
            if scenario_name not in self.scenario_data_storage:
                continue
                
            scenario_data = self.scenario_data_storage[scenario_name]
            
            # Create figure with subplots for each motor variable
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            fig.suptitle(f'Motor Performance Comparison - {scenario_name.replace("_", " ")}', fontsize=16)
            
            # Time vector for simulation
            t = scenario_data['time']
            
            # Plot real motor performance
            real_outputs = scenario_data['all_outputs']
            
            # Color scheme
            colors = {'Real': 'black', 'PSO': 'blue', 'BFO': 'red', 'Chaotic PSO-DSO': 'green'}
            
            # For each algorithm, get best parameters and simulate
            for alg_name in self.algorithms.keys():
                # Find best run for this scenario
                alg_results = [r for r in self.detailed_results[alg_name] if r['scenario'] == scenario_name]
                if alg_results:
                    best_run = min(alg_results, key=lambda x: x['error'])
                    best_params = np.array([best_run[f'identified_{name}'] for name in self.param_names])
                    
                    # Simulate with identified parameters
                    _, sim_outputs = simulate_motor(best_params, t_span=[0, 1.5], n_points=len(t))
                    
                    # Plot 1: Current Magnitude
                    axes[0].plot(t, sim_outputs['Is_mag'], label=f'{alg_name} (Error: {best_run["error"]:.1f}%)', 
                               color=colors[alg_name], linewidth=1.5, alpha=0.8)
                    
                    # Plot 2: Torque
                    axes[1].plot(t, sim_outputs['Te'], label=f'{alg_name}', 
                               color=colors[alg_name], linewidth=1.5, alpha=0.8)
                    
                    # Plot 3: Speed (RPM)
                    axes[2].plot(t, sim_outputs['rpm'], label=f'{alg_name}', 
                               color=colors[alg_name], linewidth=1.5, alpha=0.8)
            
            # Plot real motor performance
            axes[0].plot(t, real_outputs['Is_mag'], 'k-', label='Real Motor', linewidth=2)
            axes[1].plot(t, real_outputs['Te'], 'k-', label='Real Motor', linewidth=2)
            axes[2].plot(t, real_outputs['rpm'], 'k-', label='Real Motor', linewidth=2)
            
            # Formatting for Current plot
            axes[0].set_ylabel('Current Magnitude (A)')
            axes[0].set_title('Stator Current Comparison')
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim([0, 1.5])
            
            # Formatting for Torque plot
            axes[1].set_ylabel('Torque (Nm)')
            axes[1].set_title('Electromagnetic Torque Comparison')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, 1.5])
            
            # Formatting for Speed plot
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Speed (RPM)')
            axes[2].set_title('Motor Speed Comparison')
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xlim([0, 1.5])
            
            plt.tight_layout()
            
            # Save figure
            filename = f"results/plots/motor_comparison_{scenario_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved motor comparison plot: {filename}")
            
            plt.show()
    
    def plot_convergence_curves(self):
        """Plot convergence curves for algorithm comparison"""
        
        print(f"\nGenerating convergence analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Convergence Analysis', fontsize=16)
        
        colors = {'PSO': 'blue', 'BFO': 'red', 'Chaotic PSO-DSO': 'green'}
        
        # Plot 1: Average convergence curves
        ax1 = axes[0, 0]
        for alg_name in self.algorithms.keys():
            curves = self.convergence_curves[alg_name]
            if curves:
                min_length = min(len(curve) for curve in curves)
                normalized_curves = [curve[:min_length] for curve in curves]
                
                mean_curve = np.mean(normalized_curves, axis=0)
                std_curve = np.std(normalized_curves, axis=0)
                
                iterations = range(len(mean_curve))
                ax1.semilogy(iterations, mean_curve, label=alg_name, 
                            color=colors[alg_name], linewidth=2)
                ax1.fill_between(iterations, 
                               mean_curve - std_curve, 
                               mean_curve + std_curve, 
                               alpha=0.2, color=colors[alg_name])
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost Function (log scale)')
        ax1.set_title('A) Average Convergence Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best convergence curves
        ax2 = axes[0, 1]
        for alg_name in self.algorithms.keys():
            curves = self.convergence_curves[alg_name]
            if curves:
                best_curve = min(curves, key=lambda x: x[-1])
                iterations = range(len(best_curve))
                ax2.semilogy(iterations, best_curve, label=alg_name, 
                            color=colors[alg_name], linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cost Function (log scale)')
        ax2.set_title('B) Best Convergence Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence speed comparison
        ax3 = axes[1, 0]
        convergence_speeds = []
        alg_names = []
        for alg_name in self.algorithms.keys():
            valid_convergence = [c for c in self.results[alg_name]['convergence_iterations'] if c != -1]
            if valid_convergence:
                avg_convergence = np.mean(valid_convergence)
                convergence_speeds.append(avg_convergence)
                alg_names.append(alg_name)
        
        if convergence_speeds:
            bars = ax3.bar(alg_names, convergence_speeds, 
                          color=[colors[name] for name in alg_names], alpha=0.7)
            ax3.set_ylabel('Average Convergence Iteration')
            ax3.set_title('C) Convergence Speed Comparison')
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, convergence_speeds):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 4: Success rate vs iterations
        ax4 = axes[1, 1]
        for alg_name in self.algorithms.keys():
            curves = self.convergence_curves[alg_name]
            if curves:
                max_length = max(len(curve) for curve in curves)
                success_rates = []
                
                for iteration in range(0, max_length, 5):
                    success_count = 0
                    total_count = 0
                    
                    for curve in curves:
                        if iteration < len(curve):
                            if curve[iteration] < 1e-3:
                                success_count += 1
                            total_count += 1
                    
                    if total_count > 0:
                        success_rates.append(success_count / total_count * 100)
                    else:
                        success_rates.append(0)
                
                iterations = range(0, len(success_rates) * 5, 5)
                ax4.plot(iterations, success_rates, label=alg_name, 
                        color=colors[alg_name], linewidth=2, marker='o')
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('D) Success Rate Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = "results/plots/convergence_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved convergence analysis plot: {filename}")
        
        plt.show()
        
        return fig
    
    def generate_academic_plots(self):
        """Generate publication-quality plots for the paper"""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot configurations
        algorithms = list(self.results.keys())
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Parameter Error Comparison
        plt.subplot(2, 3, 1)
        error_data = [self.results[alg]['errors'] for alg in algorithms]
        box_plot = plt.boxplot(error_data, labels=algorithms, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Parameter Error (%)')
        plt.title('A) Parameter Identification Accuracy')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 2: Optimization Time Comparison  
        plt.subplot(2, 3, 2)
        time_data = [self.results[alg]['times'] for alg in algorithms]
        box_plot = plt.boxplot(time_data, labels=algorithms, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Optimization Time (s)')
        plt.title('B) Computational Efficiency')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 3: Convergence Comparison
        plt.subplot(2, 3, 3)
        for i, alg_name in enumerate(algorithms):
            curves = self.convergence_curves[alg_name]
            if curves:
                median_idx = len(curves) // 2
                representative_curve = sorted(curves, key=lambda x: x[-1])[median_idx]
                
                iterations = range(len(representative_curve))
                plt.semilogy(iterations, representative_curve, 
                           label=alg_name, color=colors[i], linewidth=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function (log scale)')
        plt.title('C) Convergence Behavior')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Success Rate Analysis
        plt.subplot(2, 3, 4)
        success_rates = []
        for alg in algorithms:
            errors = self.results[alg]['errors']
            success_rate = np.sum(np.array(errors) < 5.0) / len(errors) * 100
            success_rates.append(success_rate)
        
        bars = plt.bar(algorithms, success_rates, color=colors, alpha=0.7)
        plt.ylabel('Success Rate (%)')
        plt.title('D) Success Rate (<5% Error)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 5: Robustness vs Accuracy
        plt.subplot(2, 3, 5)
        for i, alg in enumerate(algorithms):
            errors = self.results[alg]['errors']
            robustness = np.mean(self.results[alg]['robustness_scores'])
            accuracy = 1 / (np.mean(errors) + 1)
            
            plt.scatter(accuracy, robustness, s=150, alpha=0.7, 
                       color=colors[i], label=alg, edgecolors='black')
        
        plt.xlabel('Accuracy (Inverse Error)')
        plt.ylabel('Robustness Score')
        plt.title('E) Accuracy vs Robustness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Overall Performance Radar
        plt.subplot(2, 3, 6, projection='polar')
        
        categories = ['Accuracy', 'Speed', 'Robustness']
        N = len(categories)
        
        for i, alg in enumerate(algorithms):
            errors = self.results[alg]['errors']
            times = self.results[alg]['times']
            robustness = np.mean(self.results[alg]['robustness_scores'])
            
            accuracy = 1 / (np.mean(errors) + 1)
            speed = 1 / (np.mean(times) + 0.1)
            
            max_accuracy = max([1 / (np.mean(self.results[a]['errors']) + 1) for a in algorithms])
            max_speed = max([1 / (np.mean(self.results[a]['times']) + 0.1) for a in algorithms])
            max_robustness = max([np.mean(self.results[a]['robustness_scores']) for a in algorithms])
            
            values = [accuracy/max_accuracy, speed/max_speed, robustness/max_robustness]
            
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values = values + values[:1]
            angles += angles[:1]
            
            plt.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            plt.fill(angles, values, alpha=0.25, color=colors[i])
        
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('F) Overall Performance Comparison')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save figure
        filename = "results/plots/academic_performance_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved academic performance analysis plot: {filename}")
        
        plt.show()
        
        return fig
    
    def statistical_analysis(self):
        """Perform statistical analysis of results"""
        
        print(f"\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        
        algorithms = list(self.results.keys())
        
        # ANOVA for parameter errors
        error_groups = [self.results[alg]['errors'] for alg in algorithms]
        f_stat, p_value = f_oneway(*error_groups)
        
        print(f"\nANOVA Test for Parameter Errors:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Pairwise t-tests
        print(f"\nPairwise t-tests (Parameter Errors):")
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    t_stat, p_val = ttest_ind(self.results[alg1]['errors'], 
                                            self.results[alg2]['errors'])
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"{alg1} vs {alg2}: t={t_stat:.3f}, p={p_val:.6f} {significance}")
        
        # Summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print("-" * 60)
        
        for alg in algorithms:
            errors = self.results[alg]['errors']
            times = self.results[alg]['times']
            robustness = np.mean(self.results[alg]['robustness_scores'])
            
            print(f"\n{alg}:")
            print(f"  Parameter Error: {np.mean(errors):.3f}% ± {np.std(errors):.3f}%")
            print(f"  Optimization Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
            print(f"  Robustness Score: {robustness:.3f}")
            
            success_rate = np.sum(np.array(errors) < 5.0) / len(errors) * 100
            print(f"  Success Rate (<5% error): {success_rate:.1f}%")
            
            valid_convergence = [c for c in self.results[alg]['convergence_iterations'] if c != -1]
            if valid_convergence:
                print(f"  Average Convergence Iteration: {np.mean(valid_convergence):.1f}")

# ===============================================================================
# MAIN EXECUTION WITH ENHANCED EXPORT FEATURES
# ===============================================================================

def run_conference_study():
    """Execute complete study with CSV export and motor comparison plots"""
    
    print("COMPARATIVE STUDY: Bio-Inspired Algorithms for Digital Twin Parameter Adaptation")
    print("=" * 80)
    print("Target: Mechatronics, Control & AI Conference")
    print("Enhanced with CSV Export and Motor Performance Comparison")
    print("=" * 80)
    
    # Motor parameters (2HP, 60Hz)
    ideal_motor_params = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
    
    # Initialize comparative framework
    comparator = DigitalTwinComparator(ideal_motor_params)
    
    # Execute comparative study
    print("\nPhase 1: Executing Comparative Study...")
    results = comparator.run_comparative_study(n_runs=5)  # Increase to 15 for paper
    
    # Generate motor comparison plots
    print("\nPhase 2: Generating Motor Performance Comparison Plots...")
    comparator.plot_motor_comparison()
    
    # Plot convergence curves
    print("\nPhase 3: Analyzing Convergence Behavior...")
    convergence_fig = comparator.plot_convergence_curves()
    
    # Statistical analysis
    print("\nPhase 4: Statistical Analysis...")
    comparator.statistical_analysis()
    
    # Generate academic plots
    print("\nPhase 5: Generating Publication Plots...")
    performance_fig = comparator.generate_academic_plots()
    
    # Final summary
    print(f"\n" + "="*80)
    print("STUDY COMPLETED - FILES GENERATED")
    print("="*80)
    print("\n📁 GENERATED FILES:")
    print("  CSV Files:")
    print("    • results/csv/PSO_results.csv")
    print("    • results/csv/BFO_results.csv")
    print("    • results/csv/Chaotic_PSO-DSO_results.csv")
    print("\n  PNG Plots:")
    print("    • results/plots/motor_comparison_Normal_Operation.png")
    print("    • results/plots/motor_comparison_High_Temperature.png")
    print("    • results/plots/motor_comparison_Severe_Conditions.png")
    print("    • results/plots/convergence_analysis.png")
    print("    • results/plots/academic_performance_analysis.png")
    
    algorithms = list(results.keys())
    
    # Find best performing algorithm
    best_accuracy = min(algorithms, key=lambda alg: np.mean(results[alg]['errors']))
    best_speed = min(algorithms, key=lambda alg: np.mean(results[alg]['times']))
    best_robustness = max(algorithms, key=lambda alg: np.mean(results[alg]['robustness_scores']))
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"  ✓ Best Accuracy: {best_accuracy}")
    print(f"  ✓ Best Speed: {best_speed}")  
    print(f"  ✓ Best Robustness: {best_robustness}")
    
    return comparator, results

if __name__ == "__main__":
    # Execute complete conference study
    print("Starting Enhanced Conference Study with CSV Export and Motor Comparison...")
    study_results = run_conference_study()
    
    print(f"\n🎯 STUDY COMPLETED SUCCESSFULLY")
    print("All results have been saved to the 'results' directory.")
    print("Check the 'csv' and 'plots' subdirectories for your files.")