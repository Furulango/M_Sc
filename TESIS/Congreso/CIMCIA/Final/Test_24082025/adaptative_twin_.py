# adaptive_digital_twin_system_fair.py
# ADAPTIVE DIGITAL TWIN SYSTEM: Fair Comparison Framework with Progress Indicators
# Level 1: Fixed Budget Protocol - All algorithms use same evaluation budget
# Level 3: Adaptive Convergence Protocol - Stop on convergence criteria
# For Mechatronics, Control & AI Conference Submission

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
import pickle
from functools import partial
from tqdm import tqdm # Import tqdm for progress bars

warnings.filterwarnings('ignore')

# Multiprocessing has been removed as requested for simplicity and compatibility.
# The script will now run sequentially.
print("System configured for sequential processing. Multiprocessing is disabled.")

# Create directories for results
os.makedirs('results', exist_ok=True)
os.makedirs('results/csv', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# ===============================================================================
# FAIR COMPARISON CONFIGURATION
# ===============================================================================

class FairComparisonConfig:
    """Configuration for fair comparison between algorithms"""
    
    # LEVEL 1: Fixed Budget Protocol
    CALIBRATION_BUDGET = 40000  # Total evaluations for calibration
    ADAPTATION_BUDGET = 12000   # Total evaluations for adaptation
    
    # LEVEL 3: Adaptive Convergence Protocol
    CONVERGENCE_ERROR_THRESHOLD = 0.01  # 1% error threshold
    STAGNATION_ITERATIONS = 20  # Iterations without improvement
    MIN_ITERATIONS = 10  # Minimum iterations before checking convergence
    
    @staticmethod
    def get_pso_config(is_adaptation=False):
        """Get PSO configuration for fair comparison"""
        if is_adaptation:
            # Adaptation: 12,000 evaluations
            return {
                'n_particles': 100,
                'max_iter': 120,  # 100 × 120 = 12,000
                'total_evals': 12000
            }
        else:
            # Calibration: 40,000 evaluations
            return {
                'n_particles': 125,
                'max_iter': 320,  # 125 × 320 = 40,000
                'total_evals': 40000
            }
    
    @staticmethod
    def get_bfo_config(is_adaptation=False):
        """Get BFO configuration for fair comparison"""
        if is_adaptation:
            # Adaptation: ~12,000 evaluations
            return {
                'n_bacteria': 30,
                'n_chemotactic': 50,
                'n_swim': 4,
                'n_reproductive': 2,
                'n_elimination': 2,
                'total_evals': 30 * 50 * 4 * 2 * 2  # = 12,000
            }
        else:
            # Calibration: ~40,000 evaluations
            return {
                'n_bacteria': 40,
                'n_chemotactic': 63,  
                'n_swim': 4,
                'n_reproductive': 4,
                'n_elimination': 2,
                'total_evals': 40 * 63 * 4 * 4 * 2  # = 40,320 ≈ 40,000
            }
    
    @staticmethod
    def get_chaotic_pso_config(is_adaptation=False):
        """Get Chaotic PSO-DSO configuration for fair comparison"""
        if is_adaptation:
            # Adaptation: 12,000 evaluations
            return {
                'n_particles': 80,
                'max_iter': 150,  # 80 × 150 = 12,000
                'total_evals': 12000
            }
        else:
            # Calibration: 40,000 evaluations
            return {
                'n_particles': 100,
                'max_iter': 400,  # 100 × 400 = 40,000
                'total_evals': 40000
            }

# ===============================================================================
# MOTOR SIMULATION FRAMEWORK WITH ENHANCED OBSERVABILITY
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
                        t_span, [0,0,0,0,0], dense_output=True, rtol=1e-6, atol=1e-8)
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        
        Is_mag = np.sqrt(iqs**2 + ids**2)
        Te = (3*4/4) * params[4] * (iqs*idr - ids*iqr)
        rpm = wr * 60/(2*np.pi) * 2/4
        
        # Calculate power factor and efficiency for enhanced observability
        power_factor = np.cos(np.arctan2(iqs, ids))
        
        return t, {'iqs': iqs, 'ids': ids, 'Is_mag': Is_mag, 'Te': Te, 
                   'rpm': rpm, 'wr': wr, 'power_factor': power_factor}
    
    except Exception:
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'iqs': np.ones(n_points)*1e6, 'ids': np.ones(n_points)*1e6, 
                   'Is_mag': np.ones(n_points)*1e6, 'Te': np.ones(n_points)*1e6, 
                   'rpm': np.ones(n_points)*1e6, 'wr': np.ones(n_points)*1e6,
                   'power_factor': np.ones(n_points)*1e6}

# ===============================================================================
# OBJECTIVE FUNCTION CLASSES
# ===============================================================================

class CalibrationObjective:
    """Serializable calibration objective function"""
    def __init__(self, measured_current, measured_torque, measured_speed, 
                 temperature, ideal_params, param_names, param_bounds, 
                 temp_coeffs, reference_temp):
        self.measured_current = measured_current
        self.measured_torque = measured_torque
        self.measured_speed = measured_speed
        self.temperature = temperature
        self.ideal_params = ideal_params
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.temp_coeffs = temp_coeffs
        self.reference_temp = reference_temp
        
        # Pre-calculate scales
        self.current_scale = np.max(np.abs(measured_current))
        self.torque_scale = np.max(np.abs(measured_torque))
        self.speed_scale = np.max(np.abs(measured_speed))
    
    def compensate_temperature(self, params):
        """Apply temperature compensation"""
        temp_diff = self.temperature - self.reference_temp
        compensated = params.copy()
        
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        
        return compensated
    
    def __call__(self, candidate_params):
        """Evaluate objective function"""
        try:
            penalty = 0
            
            # Check parameter bounds
            for i, (param, param_name) in enumerate(zip(candidate_params, self.param_names)):
                min_val, max_val = self.param_bounds[param_name]
                if param < min_val or param > max_val:
                    penalty += 1e6 * (1 + abs(param - np.clip(param, min_val, max_val)))
            
            # Check physical relationships
            Ls = candidate_params[2] + candidate_params[4]
            Lr = candidate_params[3] + candidate_params[4]
            if Ls <= candidate_params[2] or Lr <= candidate_params[3]:
                penalty += 1e6
            
            if candidate_params[0] <= 0 or candidate_params[1] <= 0:
                penalty += 1e8
            
            # Apply temperature compensation
            temp_compensated = self.compensate_temperature(candidate_params)
            
            # Simulate with candidate parameters
            _, sim_outputs = simulate_motor(temp_compensated, 
                                          t_span=[0, len(self.measured_current)*0.005],
                                          n_points=len(self.measured_current))
            
            sim_current = sim_outputs['Is_mag']
            sim_torque = sim_outputs['Te']
            sim_speed = sim_outputs['rpm']
            
            # Normalized MSE for each signal
            current_mse = np.mean(((self.measured_current - sim_current) / self.current_scale)**2)
            torque_mse = np.mean(((self.measured_torque - sim_torque) / self.torque_scale)**2)
            speed_mse = np.mean(((self.measured_speed - sim_speed) / self.speed_scale)**2)
            
            # Weighted combination - ALL SIGNALS
            weights = {'current': 0.5, 'torque': 0.3, 'speed': 0.2}
            total_mse = (weights['current'] * current_mse + 
                         weights['torque'] * torque_mse + 
                         weights['speed'] * speed_mse)
            
            # Regularization
            regularization = 0
            for i, param in enumerate(candidate_params):
                expected = self.ideal_params[i]
                regularization += 0.001 * ((param - expected) / expected)**2
            
            return total_mse + penalty + regularization
            
        except Exception:
            return 1e10

class AdaptationObjective:
    """Serializable adaptation objective function"""
    def __init__(self, measured_current, temperature, param_names, param_bounds,
                 temp_coeffs, reference_temp):
        self.measured_current = measured_current
        self.temperature = temperature
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.temp_coeffs = temp_coeffs
        self.reference_temp = reference_temp
        
        # Pre-calculate scale
        self.current_scale = np.max(np.abs(measured_current))
    
    def compensate_temperature(self, params):
        """Apply temperature compensation"""
        temp_diff = self.temperature - self.reference_temp
        compensated = params.copy()
        
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        
        return compensated
    
    def __call__(self, candidate_params):
        """Evaluate objective function"""
        try:
            penalty = 0
            
            # Check parameter bounds
            for i, (param, param_name) in enumerate(zip(candidate_params, self.param_names)):
                min_val, max_val = self.param_bounds[param_name]
                if param < min_val or param > max_val:
                    penalty += 1e6 * (1 + abs(param - np.clip(param, min_val, max_val)))
            
            # Check physical relationships
            Ls = candidate_params[2] + candidate_params[4]
            Lr = candidate_params[3] + candidate_params[4]
            if Ls <= candidate_params[2] or Lr <= candidate_params[3]:
                penalty += 1e6
            
            if candidate_params[0] <= 0 or candidate_params[1] <= 0:
                penalty += 1e8
            
            # Apply temperature compensation
            temp_compensated = self.compensate_temperature(candidate_params)
            
            # Simulate with candidate parameters
            _, sim_outputs = simulate_motor(temp_compensated, 
                                          t_span=[0, len(self.measured_current)*0.005],
                                          n_points=len(self.measured_current))
            
            sim_current = sim_outputs['Is_mag']
            
            # MSE for current only
            current_mse = np.mean(((self.measured_current - sim_current) / self.current_scale)**2)
            
            return current_mse + penalty
            
        except Exception:
            return 1e10

# ===============================================================================
# ENHANCED ADAPTIVE BIO-INSPIRED ALGORITHMS WITH FAIR COMPARISON
# ===============================================================================

class EnhancedAdaptivePSO:
    """Enhanced Adaptive PSO with fair comparison and convergence detection"""
    
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False, 
                 comparison_level=1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        self.comparison_level = comparison_level  # 1: Fixed Budget, 3: Adaptive Convergence
        
        # Get fair configuration
        config = FairComparisonConfig.get_pso_config(is_adaptation)
        self.n_particles = config['n_particles']
        self.max_iter = config['max_iter']
        self.budget = config['total_evals']
        
        self.n_dims = len(bounds[0])
        
        # Tracking metrics
        self.cost_history = []
        self.evaluation_count = 0
        self.convergence_iteration = -1
        self.convergence_evaluation = -1
        self.best_cost = float('inf')
        self.best_params = None
        self.stagnation_counter = 0
        self.last_best_cost = float('inf')
        
        # Adaptive PySwarms options
        if is_adaptation:
            self.options = {'c1': 1.5, 'c2': 2.5, 'w': 0.7, 'k': 5, 'p': 2}
        else:
            self.options = {'c1': 2.05, 'c2': 2.05, 'w': 0.9, 'k': 5, 'p': 2}
    
    def check_convergence(self):
        """Check convergence criteria for Level 3"""
        if self.comparison_level != 3:
            return False
        
        # Check if minimum iterations passed
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        
        # Check error threshold
        if self.best_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        
        # Check stagnation
        if abs(self.best_cost - self.last_best_cost) < 1e-8:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_cost = self.best_cost
        
        if self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS:
            return True
        
        return False
    
    def smart_initialization(self):
        """Smart initialization for particles"""
        init_pos = np.zeros((self.n_particles, self.n_dims))
        
        if self.is_adaptation and self.base_params is not None:
            # ADAPTATION MODE: Initialize around base parameters
            n_close = int(self.n_particles * 0.6)
            for i in range(n_close):
                init_pos[i] = self.base_params * np.random.uniform(0.95, 1.05, self.n_dims)
            
            for i in range(n_close, self.n_particles):
                init_pos[i] = self.base_params * np.random.uniform(0.9, 1.1, self.n_dims)
                
        elif self.base_params is not None:
            # CALIBRATION MODE: Use ideal parameters
            n_near = int(self.n_particles * 0.4)
            for i in range(n_near):
                init_pos[i] = self.base_params * np.random.uniform(0.8, 1.2, self.n_dims)
            
            n_medium = int(self.n_particles * 0.3)
            for i in range(n_near, n_near + n_medium):
                init_pos[i] = self.base_params * np.random.uniform(0.7, 1.3, self.n_dims)
            
            for i in range(n_near + n_medium, self.n_particles):
                init_pos[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dims)
        else:
            init_pos = np.random.uniform(self.bounds[0], self.bounds[1], 
                                         (self.n_particles, self.n_dims))
        
        return np.clip(init_pos, self.bounds[0], self.bounds[1])
    
    def pso_objective_wrapper(self, x):
        """Wrapper for PySwarms objective function with evaluation counting"""
        costs = np.array([self.objective_func(particle) for particle in x])
        self.evaluation_count += len(x)
        
        # Track best
        current_best_idx = np.argmin(costs)
        current_best = costs[current_best_idx]
        
        if current_best < self.best_cost:
            self.best_cost = current_best
            self.best_params = x[current_best_idx].copy()
            if self.convergence_iteration == -1 and current_best < 1e-6:
                self.convergence_iteration = len(self.cost_history)
                self.convergence_evaluation = self.evaluation_count
        
        self.cost_history.append(current_best)
        
        return costs
    
    def optimize(self):
        """Execute optimization with fair comparison"""
        start_time = time.time()
        
        mode = "Adaptation" if self.is_adaptation else "Calibration"
        level_str = f"Level {self.comparison_level}"
        
        print(f"    Enhanced PSO ({mode}, {level_str}): {self.n_particles} particles, "
              f"max {self.max_iter} iter, budget {self.budget} evals")
        
        # Initialize
        init_pos = self.smart_initialization()
        
        # Create optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.n_dims,
            options=self.options,
            bounds=self.bounds,
            init_pos=init_pos
        )
        
        # Perform optimization
        # MODIFICATION: Changed verbose to an integer to show progress periodically
        # It will print progress 10 times during the optimization run.
        progress_reports = 10
        verbose_freq = max(1, self.max_iter // progress_reports)
        best_cost, best_pos = optimizer.optimize(
            self.pso_objective_wrapper, 
            iters=self.max_iter,
            verbose=verbose_freq
        )
        
        optimization_time = time.time() - start_time
        
        print(f"    Enhanced PSO ({mode}): Completed - Best cost: {self.best_cost:.2e}, "
              f"Evals: {self.evaluation_count}, Time: {optimization_time:.2f}s")
        
        return self.best_cost, self.best_params, optimization_time, self.evaluation_count

class EnhancedAdaptiveBFO:
    """Enhanced Adaptive BFO with fair comparison and convergence detection"""
    
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False,
                 comparison_level=1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        self.comparison_level = comparison_level
        
        # Get fair configuration
        config = FairComparisonConfig.get_bfo_config(is_adaptation)
        self.S = config['n_bacteria']
        self.Nc = config['n_chemotactic']
        self.Ns = config['n_swim']
        self.Nre = config['n_reproductive']
        self.Ned = config['n_elimination']
        self.budget = config['total_evals']
        
        self.Ped = 0.2 if not is_adaptation else 0.1
        self.Ci = 0.05 if not is_adaptation else 0.02
        
        self.n_dims = len(bounds[0])
        self.lb, self.ub = bounds
        
        # Initialize bacteria
        self.bacteria = self._smart_initialization()
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.evaluation_count = self.S
        self.health = np.zeros(self.S)
        
        self.best_pos = self.bacteria[np.argmin(self.costs)]
        self.best_cost = np.min(self.costs)
        self.best_params = self.best_pos.copy()
        
        # Tracking
        self.cost_history = [self.best_cost]
        self.convergence_iteration = -1
        self.convergence_evaluation = -1
        self.stagnation_counter = 0
        self.last_best_cost = self.best_cost
    
    def check_convergence(self):
        """Check convergence criteria for Level 3"""
        if self.comparison_level != 3:
            return False
        
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        
        if self.best_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        
        if abs(self.best_cost - self.last_best_cost) < 1e-8:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_cost = self.best_cost
        
        if self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS:
            return True
        
        return False
    
    def _smart_initialization(self):
        """Smart initialization for bacteria"""
        bacteria = np.zeros((self.S, self.n_dims))
        
        if self.is_adaptation and self.base_params is not None:
            n_close = int(self.S * 0.7)
            for i in range(n_close):
                bacteria[i] = self.base_params * np.random.uniform(0.95, 1.05, self.n_dims)
            
            for i in range(n_close, self.S):
                bacteria[i] = self.base_params * np.random.uniform(0.9, 1.1, self.n_dims)
                
        elif self.base_params is not None:
            n_near = int(self.S * 0.4)
            for i in range(n_near):
                bacteria[i] = self.base_params * np.random.uniform(0.8, 1.2, self.n_dims)
            
            for i in range(n_near, self.S):
                bacteria[i] = np.random.uniform(self.lb, self.ub, self.n_dims)
        else:
            bacteria = np.random.uniform(self.lb, self.ub, (self.S, self.n_dims))
        
        return np.clip(bacteria, self.lb, self.ub)
    
    def optimize(self):
        """Execute BFO optimization with fair comparison"""
        start_time = time.time()
        
        mode = "Adaptation" if self.is_adaptation else "Calibration"
        level_str = f"Level {self.comparison_level}"
        
        print(f"    Enhanced BFO ({mode}, {level_str}): {self.S} bacteria, "
              f"budget {self.budget} evals")

        # MODIFICATION: Added tqdm progress bar
        total_chemo_steps = self.Ned * self.Nre * self.Nc
        pbar = tqdm(total=total_chemo_steps, desc=f"BFO ({mode})", leave=False)

        for l in range(self.Ned):
            for k in range(self.Nre):
                for j in range(self.Nc):
                    self._update_best()
                    
                    # Check convergence for Level 3
                    if self.comparison_level == 3 and self.check_convergence():
                        pbar.close()
                        print(f"    BFO converged at chemotactic step {pbar.n}, "
                              f"evaluations: {self.evaluation_count}")
                        optimization_time = time.time() - start_time
                        return self.best_cost, self.best_pos, optimization_time, self.evaluation_count
                    
                    last_costs = np.copy(self.costs)
                    directions = self._tumble()
                    
                    adaptive_step = self.Ci * (1 - j/self.Nc * 0.5)
                    
                    # Swimming phase
                    for m in range(self.Ns):
                        new_pos = self.bacteria + adaptive_step * directions
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_costs = np.array([self.objective_func(p) for p in new_pos])
                        self.evaluation_count += self.S
                        
                        improved_mask = new_costs < self.costs
                        self.bacteria[improved_mask] = new_pos[improved_mask]
                        self.costs[improved_mask] = new_costs[improved_mask]
                        self.health += last_costs - self.costs
                        
                        if not np.any(improved_mask):
                            break
                    
                    self.cost_history.append(self.best_cost)
                    pbar.update(1) # Update progress bar
                    pbar.set_postfix(best_cost=f'{self.best_cost:.2e}')

                self._reproduce()
            self._eliminate_disperse()
        
        pbar.close() # Close progress bar
        self._update_best()
        optimization_time = time.time() - start_time
        
        print(f"    Enhanced BFO ({mode}): Completed - Best cost: {self.best_cost:.2e}, "
              f"Evals: {self.evaluation_count}, Time: {optimization_time:.2f}s")
        
        return self.best_cost, self.best_pos, optimization_time, self.evaluation_count
    
    def _update_best(self):
        min_cost_idx = np.argmin(self.costs)
        if self.costs[min_cost_idx] < self.best_cost:
            self.best_cost = self.costs[min_cost_idx]
            self.best_pos = self.bacteria[min_cost_idx]
            self.best_params = self.best_pos.copy()
    
    def _tumble(self):
        direction = np.random.uniform(-1, 1, (self.S, self.n_dims))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        return direction / norm
    
    def _reproduce(self):
        sorted_indices = np.argsort(self.health)
        n_survive = self.S // 2
        
        survivors_pos = self.bacteria[sorted_indices[:n_survive]]
        
        offspring = survivors_pos + np.random.normal(0, 0.01 if not self.is_adaptation else 0.005, 
                                                     survivors_pos.shape)
        offspring = np.clip(offspring, self.lb, self.ub)
        
        self.bacteria = np.concatenate([survivors_pos, offspring])
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.evaluation_count += self.S
        self.health = np.zeros(self.S)
    
    def _eliminate_disperse(self):
        for i in range(self.S):
            if np.random.rand() < self.Ped:
                if self.is_adaptation and self.base_params is not None:
                    self.bacteria[i] = self.best_pos * np.random.uniform(0.95, 1.05, self.n_dims)
                elif np.random.rand() < 0.3 and self.base_params is not None:
                    self.bacteria[i] = self.best_pos * np.random.uniform(0.9, 1.1, self.n_dims)
                else:
                    self.bacteria[i] = np.random.uniform(self.lb, self.ub, self.n_dims)
                self.bacteria[i] = np.clip(self.bacteria[i], self.lb, self.ub)
                self.costs[i] = self.objective_func(self.bacteria[i])
                self.evaluation_count += 1

class EnhancedAdaptiveChaoticPSODSO:
    """Enhanced Adaptive Chaotic PSO-DSO with fair comparison"""
    
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False,
                 comparison_level=1):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        self.comparison_level = comparison_level
        
        # Get fair configuration
        config = FairComparisonConfig.get_chaotic_pso_config(is_adaptation)
        self.n_particles = config['n_particles']
        self.max_iter = config['max_iter']
        self.budget = config['total_evals']
        
        self.n_dims = len(bounds[0])
        
        # Dynamic parameters
        if is_adaptation:
            self.w_max = 0.7
            self.w_min = 0.4
            self.c1_init = 2.0
            self.c2_init = 1.0
        else:
            self.w_max = 0.95
            self.w_min = 0.3
            self.c1_init = 2.8
            self.c2_init = 0.3
        
        # Chaotic maps
        self.chaos_param = 4.0
        self.chaos_values = np.random.rand(self.n_particles)
        
        # Initialize particles
        self.particles = self._smart_initialization()
        self.velocities = np.zeros((self.n_particles, self.n_dims))
        self.pbest = self.particles.copy()
        self.pbest_costs = np.array([objective_func(p) for p in self.particles])
        self.gbest = self.pbest[np.argmin(self.pbest_costs)]
        self.gbest_cost = np.min(self.pbest_costs)
        self.best_params = self.gbest.copy()
        
        # Tracking
        self.evaluation_count = self.n_particles
        self.cost_history = [self.gbest_cost]
        self.convergence_iteration = -1
        self.convergence_evaluation = -1
        self.stagnation_counter = 0
        self.last_best_cost = self.gbest_cost
    
    def check_convergence(self):
        """Check convergence criteria for Level 3"""
        if self.comparison_level != 3:
            return False
        
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        
        if self.gbest_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        
        if abs(self.gbest_cost - self.last_best_cost) < 1e-8:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_cost = self.gbest_cost
        
        if self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS:
            return True
        
        return False
    
    def _smart_initialization(self):
        """Smart initialization for particles"""
        particles = np.zeros((self.n_particles, self.n_dims))
        
        if self.is_adaptation and self.base_params is not None:
            n_close = int(self.n_particles * 0.6)
            for i in range(n_close):
                particles[i] = self.base_params * np.random.uniform(0.95, 1.05, self.n_dims)
            
            for i in range(n_close, self.n_particles):
                particles[i] = self.base_params * np.random.uniform(0.9, 1.1, self.n_dims)
                
        elif self.base_params is not None:
            n_near = int(self.n_particles * 0.4)
            for i in range(n_near):
                particles[i] = self.base_params * np.random.uniform(0.8, 1.2, self.n_dims)
            
            n_medium = int(self.n_particles * 0.3)
            for i in range(n_near, n_near + n_medium):
                particles[i] = self.base_params * np.random.uniform(0.7, 1.3, self.n_dims)
            
            for i in range(n_near + n_medium, self.n_particles):
                particles[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.n_dims)
        else:
            particles = np.random.uniform(self.bounds[0], self.bounds[1], 
                                          (self.n_particles, self.n_dims))
        
        return np.clip(particles, self.bounds[0], self.bounds[1])
    
    def optimize(self):
        """Execute Chaotic PSO-DSO optimization with fair comparison"""
        start_time = time.time()
        
        mode = "Adaptation" if self.is_adaptation else "Calibration"
        level_str = f"Level {self.comparison_level}"
        
        print(f"    Enhanced Chaotic PSO-DSO ({mode}, {level_str}): {self.n_particles} particles, "
              f"max {self.max_iter} iter, budget {self.budget} evals")
        
        stagnation_counter = 0
        last_best_cost = self.gbest_cost
        
        # MODIFICATION: Added tqdm progress bar for the main loop
        pbar = tqdm(range(self.max_iter), desc=f"Chaotic PSO-DSO ({mode})", leave=False)
        for iteration in pbar:
            # Check convergence for Level 3
            if self.comparison_level == 3 and self.check_convergence():
                print(f"    Chaotic PSO-DSO converged at iteration {iteration}, "
                      f"evaluations: {self.evaluation_count}")
                break
            
            # Dynamic parameter adjustment
            progress_ratio = iteration / self.max_iter
            w = self.w_max - (self.w_max - self.w_min) * progress_ratio
            c1 = self.c1_init - (self.c1_init - 2.0) * progress_ratio
            c2 = self.c2_init + (2.0 - self.c2_init) * progress_ratio
            
            # Stagnation detection
            if abs(self.gbest_cost - last_best_cost) < 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_best_cost = self.gbest_cost
            
            # Chaos intensity
            chaos_intensity = 0.05 if self.is_adaptation else 0.1
            if stagnation_counter > 5:
                chaos_intensity *= 3
            
            for i in range(self.n_particles):
                # Update chaotic value
                if self.chaos_values[i] < 0.5:
                    self.chaos_values[i] = 2 * self.chaos_values[i]
                else:
                    self.chaos_values[i] = 2 * (1 - self.chaos_values[i])
                
                # Chaotic velocity update
                r1 = self.chaos_values[i]
                r2 = np.random.rand()
                
                chaos_factor = chaos_intensity * self.chaos_values[i]
                
                # Velocity clamping
                v_max = 0.1 if self.is_adaptation else 0.2
                v_max *= (self.bounds[1] - self.bounds[0])
                
                self.velocities[i] = (w * self.velocities[i] + 
                                      c1 * r1 * (self.pbest[i] - self.particles[i]) +
                                      c2 * r2 * (self.gbest - self.particles[i]) +
                                      chaos_factor * (np.random.rand(self.n_dims) - 0.5))
                
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
                
                # Update position
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.bounds[0], self.bounds[1])
                
                # Evaluate fitness
                cost = self.objective_func(self.particles[i])
                self.evaluation_count += 1
                
                # Update personal best
                if cost < self.pbest_costs[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_costs[i] = cost
                    
                    # Update global best
                    if cost < self.gbest_cost:
                        self.gbest = self.particles[i].copy()
                        self.gbest_cost = cost
                        self.best_params = self.gbest.copy()
                        if self.convergence_iteration == -1 and cost < 1e-6:
                            self.convergence_iteration = iteration
                            self.convergence_evaluation = self.evaluation_count
            
            self.cost_history.append(self.gbest_cost)
            pbar.set_postfix(best_cost=f'{self.gbest_cost:.2e}')
        
        pbar.close() # Close the progress bar
        optimization_time = time.time() - start_time
        
        print(f"    Enhanced Chaotic PSO-DSO ({mode}): Completed - Best cost: {self.gbest_cost:.2e}, "
              f"Evals: {self.evaluation_count}, Time: {optimization_time:.2f}s")
        
        return self.gbest_cost, self.gbest, optimization_time, self.evaluation_count

# ===============================================================================
# SEQUENTIAL PROCESSING UTILITY
# ===============================================================================

def run_single_optimization(args):
    """Function to run a single optimization sequentially"""
    (alg_name, AlgorithmClass, objective, bounds, base_params, 
     is_adaptation, comparison_level, run_num, scenario_name) = args
    
    # Set random seed for reproducibility
    np.random.seed(run_num * 100 + hash(alg_name + scenario_name) % 1000)
    
    # Create and run algorithm
    algorithm = AlgorithmClass(objective, bounds, base_params, 
                               is_adaptation, comparison_level)
    cost, params, opt_time, eval_count = algorithm.optimize()
    
    # Return results
    return {
        'algorithm': alg_name,
        'run': run_num,
        'cost': cost,
        'params': params,
        'time': opt_time,
        'evaluations': eval_count,
        'cost_history': algorithm.cost_history,
        'convergence_iteration': getattr(algorithm, 'convergence_iteration', -1),
        'convergence_evaluation': getattr(algorithm, 'convergence_evaluation', -1)
    }

# ===============================================================================
# ENHANCED ADAPTIVE DIGITAL TWIN FRAMEWORK
# ===============================================================================

class EnhancedAdaptiveDigitalTwinSystem:
    """Enhanced Adaptive Digital Twin System with Fair Comparison"""
    
    def __init__(self, ideal_params, comparison_level=1):
        self.ideal_params = np.array(ideal_params)
        self.comparison_level = comparison_level  # 1: Fixed Budget, 3: Adaptive Convergence
        
        print(f"System initialized for sequential processing.")
        
        # Parameter names and physical bounds
        self.param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
        self.param_bounds = {
            'rs': (0.5, 10.0),
            'rr': (0.5, 10.0),
            'Lls': (0.001, 0.05),
            'Llr': (0.001, 0.05),
            'Lm': (0.05, 0.5),
            'J': (0.001, 0.1),
            'B': (0.0001, 0.01)
        }
        
        # Temperature compensation
        self.temp_coeffs = np.array([0.004, 0.004, 0.001, 0.001, 0.0005, 0, 0])
        self.reference_temp = 20.0
        
        # Algorithm classes (enhanced versions)
        self.algorithms = {
            'PSO': EnhancedAdaptivePSO,
            'BFO': EnhancedAdaptiveBFO,
            'Chaotic PSO-DSO': EnhancedAdaptiveChaoticPSODSO
        }
        
        # Storage
        self.digital_twin_base = {}
        self.best_calibration_error = {}
        self.results = {}
        self.convergence_curves = {}
        self.detailed_results = {}
        self.scenario_data_storage = {}
        self.evaluation_counts = {}
    
    def compensate_temperature(self, params, temperature):
        """Apply temperature compensation"""
        temp_diff = temperature - self.reference_temp
        compensated = params.copy()
        
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        
        return compensated
    
    def generate_nonideal_scenario(self, scenario_name, temperature, degradation_factor, noise_level):
        """Generate non-ideal motor scenario"""
        
        nonideal_params = self.ideal_params.copy()
        nonideal_params = self.compensate_temperature(nonideal_params, temperature)
        
        np.random.seed(hash(scenario_name) % 2**32)
        degradation = np.random.normal(1.0, degradation_factor, len(nonideal_params))
        degradation = np.clip(degradation, 0.85, 1.15)
        nonideal_params *= degradation
        
        t, outputs = simulate_motor(nonideal_params, t_span=[0, 2.0], n_points=400)
        
        current_clean = outputs['Is_mag']
        torque_clean = outputs['Te']
        speed_clean = outputs['rpm']
        
        current_noise = np.random.normal(0, noise_level * np.std(current_clean), len(current_clean))
        torque_noise = np.random.normal(0, noise_level * 0.5 * np.std(torque_clean), len(torque_clean))
        speed_noise = np.random.normal(0, noise_level * 0.3 * np.std(speed_clean), len(speed_clean))
        
        current_measured = current_clean + current_noise
        torque_measured = torque_clean + torque_noise
        speed_measured = speed_clean + speed_noise
        
        return {
            'true_params': nonideal_params,
            'measured_current': current_measured,
            'measured_torque': torque_measured,
            'measured_speed': speed_measured,
            'temperature': temperature,
            'time': t,
            'all_outputs': outputs
        }
    
    def create_calibration_objective(self, measured_current, measured_torque, 
                                     measured_speed, temperature):
        """Create serializable calibration objective"""
        return CalibrationObjective(
            measured_current, measured_torque, measured_speed,
            temperature, self.ideal_params, self.param_names,
            self.param_bounds, self.temp_coeffs, self.reference_temp
        )
    
    def create_adaptation_objective(self, measured_current, temperature):
        """Create serializable adaptation objective"""
        return AdaptationObjective(
            measured_current, temperature, self.param_names,
            self.param_bounds, self.temp_coeffs, self.reference_temp
        )
    
    def run_sequential_optimizations(self, algorithm_tasks):
        """Run multiple optimizations sequentially with a progress bar"""
        results = []
        # MODIFICATION: Added tqdm progress bar for the main task loop
        for task in tqdm(algorithm_tasks, desc="  Overall Progress"):
            results.append(run_single_optimization(task))
        return results
    
    def run_enhanced_adaptive_study(self, n_runs=10):
        """Execute enhanced adaptive digital twin study with fair comparison"""
        
        print("="*80)
        print("ENHANCED ADAPTIVE DIGITAL TWIN SYSTEM - FAIR COMPARISON FRAMEWORK")
        print(f"Comparison Level: {self.comparison_level}")
        print(f"Level 1: Fixed Budget - All algorithms use same evaluation budget")
        print(f"Level 3: Adaptive Convergence - Stop on convergence criteria")
        print(f"Processing mode: Sequential")
        print("="*80)
        
        # Test scenarios
        scenarios = [
            {'name': 'Normal_Operation', 'temp': 40, 'degradation': 0.03, 'noise': 0.01, 'phase': 1},
            {'name': 'High_Temperature', 'temp': 70, 'degradation': 0.06, 'noise': 0.02, 'phase': 2},
            {'name': 'Severe_Conditions', 'temp': 85, 'degradation': 0.10, 'noise': 0.03, 'phase': 2}
        ]
        
        # Initialize results storage
        for algorithm in self.algorithms.keys():
            self.results[algorithm] = {
                'costs': [], 'errors': [], 'times': [], 'evaluations': [],
                'convergence_iterations': [], 'convergence_evaluations': [],
                'robustness_scores': []
            }
            self.convergence_curves[algorithm] = []
            self.detailed_results[algorithm] = []
            self.digital_twin_base[algorithm] = None
            self.best_calibration_error[algorithm] = float('inf')
            self.evaluation_counts[algorithm] = []
        
        # Process scenarios
        for scenario in scenarios:
            print(f"\n" + "="*60)
            if scenario['phase'] == 1:
                print(f"PHASE 1 - CALIBRATION: {scenario['name']}")
                print(f"Budget: {FairComparisonConfig.CALIBRATION_BUDGET} evaluations")
            else:
                print(f"PHASE 2 - ADAPTATION: {scenario['name']}")
                print(f"Budget: {FairComparisonConfig.ADAPTATION_BUDGET} evaluations")
            print("="*60)
            
            # Generate scenario data
            scenario_data = self.generate_nonideal_scenario(
                scenario['name'], scenario['temp'], 
                scenario['degradation'], scenario['noise']
            )
            
            self.scenario_data_storage[scenario['name']] = scenario_data
            
            # Prepare tasks for sequential execution
            all_tasks = []
            
            for alg_name, AlgorithmClass in self.algorithms.items():
                for run in range(n_runs):
                    if scenario['phase'] == 1:
                        # Calibration
                        objective = self.create_calibration_objective(
                            scenario_data['measured_current'],
                            scenario_data['measured_torque'],
                            scenario_data['measured_speed'],
                            scenario_data['temperature']
                        )
                        
                        search_factor = 0.2
                        bounds = (self.ideal_params * (1 - search_factor), 
                                  self.ideal_params * (1 + search_factor))
                        base_params = self.ideal_params
                        is_adaptation = False
                        
                    else:
                        # Adaptation
                        if self.digital_twin_base[alg_name] is None:
                            continue
                        
                        objective = self.create_adaptation_objective(
                            scenario_data['measured_current'],
                            scenario_data['temperature']
                        )
                        
                        search_factor = 0.2
                        base = self.digital_twin_base[alg_name]
                        bounds = (base * (1 - search_factor), 
                                  base * (1 + search_factor))
                        base_params = base
                        is_adaptation = True
                    
                    task = (alg_name, AlgorithmClass, objective, bounds, 
                            base_params, is_adaptation, self.comparison_level, 
                            run, scenario['name'])
                    all_tasks.append(task)
            
            # Execute optimizations
            print(f"\n  Executing {len(all_tasks)} optimization tasks sequentially...")
            optimization_results = self.run_sequential_optimizations(all_tasks)
            
            # Process results
            for result in optimization_results:
                alg_name = result['algorithm']
                params = result['params']
                
                # Calculate error
                param_errors = np.abs((params - scenario_data['true_params']) / 
                                      scenario_data['true_params']) * 100
                param_error = np.mean(param_errors)
                
                # Update best base for calibration
                if scenario['phase'] == 1 and param_error < self.best_calibration_error[alg_name]:
                    self.digital_twin_base[alg_name] = params.copy()
                    self.best_calibration_error[alg_name] = param_error
                    print(f"    🏆 NEW BEST for {alg_name}: {param_error:.2f}%")
                
                # Store detailed results
                detailed_run = {
                    'scenario': scenario['name'], 'phase': scenario['phase'],
                    'run': result['run'] + 1, 'cost': result['cost'],
                    'error': param_error, 'time': result['time'],
                    'evaluations': result['evaluations'],
                    'convergence_iteration': result['convergence_iteration'],
                    'convergence_evaluation': result['convergence_evaluation'],
                    'identified_params': params.tolist(),
                    'true_params': scenario_data['true_params'].tolist(),
                    'is_best_base': (scenario['phase'] == 1 and 
                                   param_error == self.best_calibration_error[alg_name]),
                    'comparison_level': self.comparison_level
                }
                
                # Add individual parameter results
                for i, param_name in enumerate(self.param_names):
                    detailed_run[f'identified_{param_name}'] = params[i]
                    detailed_run[f'true_{param_name}'] = scenario_data['true_params'][i]
                    detailed_run[f'percent_error_{param_name}'] = param_errors[i]
                
                self.detailed_results[alg_name].append(detailed_run)
                
                # Store aggregated results
                self.results[alg_name]['costs'].append(result['cost'])
                self.results[alg_name]['errors'].append(param_error)
                self.results[alg_name]['times'].append(result['time'])
                self.results[alg_name]['evaluations'].append(result['evaluations'])
                self.results[alg_name]['convergence_iterations'].append(result['convergence_iteration'])
                self.results[alg_name]['convergence_evaluations'].append(result['convergence_evaluation'])
                self.convergence_curves[alg_name].append(result['cost_history'])
            
            # Print scenario summary
            for alg_name in self.algorithms.keys():
                scenario_results = [r for r in self.detailed_results[alg_name] 
                                    if r['scenario'] == scenario['name']]
                if scenario_results:
                    errors = [r['error'] for r in scenario_results]
                    times = [r['time'] for r in scenario_results]
                    evals = [r['evaluations'] for r in scenario_results]
                    
                    success_rate = np.sum(np.array(errors) < 5.0) / len(errors) * 100
                    
                    print(f"\n  {alg_name} Summary:")
                    print(f"    Mean Error: {np.mean(errors):.2f}% ± {np.std(errors):.2f}%")
                    print(f"    Mean Time: {np.mean(times):.3f}s")
                    print(f"    Mean Evaluations: {np.mean(evals):.0f}")
                    print(f"    Success Rate (<5%): {success_rate:.1f}%")
        
        # Export results
        self.export_enhanced_to_csv()
        
        return self.results
    
    def export_enhanced_to_csv(self):
        """Export enhanced results to CSV files"""
        print("\n" + "="*80)
        print("EXPORTING ENHANCED RESULTS TO CSV")
        print("="*80)
        
        for alg_name, results in self.detailed_results.items():
            if results:
                df = pd.DataFrame(results)
                
                # Reorder columns
                base_cols = ['scenario', 'phase', 'run', 'comparison_level', 
                             'cost', 'error', 'time', 'evaluations', 
                             'convergence_iteration', 'convergence_evaluation', 'is_best_base']
                param_cols = [f'identified_{name}' for name in self.param_names]
                true_cols = [f'true_{name}' for name in self.param_names]
                error_cols = [f'percent_error_{name}' for name in self.param_names]
                
                ordered_cols = base_cols + param_cols + true_cols + error_cols
                available_cols = [col for col in ordered_cols if col in df.columns]
                df = df[available_cols]
                
                # Save to CSV
                filename = f"results/csv/{alg_name.replace(' ', '_')}_enhanced_fair_L{self.comparison_level}.csv"
                df.to_csv(filename, index=False, float_format='%.6f')
                print(f"  ✓ Saved {alg_name} results to {filename}")
                
                # Print statistics
                for phase in [1, 2]:
                    phase_df = df[df['phase'] == phase]
                    if not phase_df.empty:
                        phase_name = "Calibration" if phase == 1 else "Adaptation"
                        print(f"    Phase {phase} ({phase_name}):")
                        print(f"      - Mean error: {phase_df['error'].mean():.2f}%")
                        print(f"      - Mean time: {phase_df['time'].mean():.3f}s")
                        print(f"      - Mean evaluations: {phase_df['evaluations'].mean():.0f}")
                        
                        if self.comparison_level == 3:
                            converged = phase_df[phase_df['convergence_evaluation'] > 0]
                            if not converged.empty:
                                print(f"      - Convergence rate: {len(converged)/len(phase_df)*100:.1f}%")
                                print(f"      - Mean convergence evals: {converged['convergence_evaluation'].mean():.0f}")
    
    def statistical_analysis_enhanced(self):
        """Enhanced statistical analysis with evaluation metrics"""
        print(f"\n" + "="*80)
        print("ENHANCED STATISTICAL ANALYSIS - FAIR COMPARISON")
        print(f"Comparison Level: {self.comparison_level}")
        print("="*80)
        
        algorithms = list(self.results.keys())
        
        # Analysis by phase
        for phase in [1, 2]:
            phase_name = "CALIBRATION" if phase == 1 else "ADAPTATION"
            print(f"\n{'='*60}")
            print(f"PHASE {phase}: {phase_name}")
            print('='*60)
            
            # Get phase-specific data
            phase_data = {}
            for alg in algorithms:
                phase_data[alg] = [r for r in self.detailed_results[alg] 
                                   if r['phase'] == phase]
            
            # ANOVA for errors
            if all(len(data) > 0 for data in phase_data.values()):
                error_groups = [[r['error'] for r in phase_data[alg]] for alg in algorithms]
                f_stat, p_value = f_oneway(*error_groups)
                
                print(f"\nANOVA Test for Parameter Errors:")
                print(f"F-statistic: {f_stat:.4f}")
                print(f"P-value: {p_value:.6f}")
                print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
            # Algorithm comparison
            print(f"\nALGORITHM COMPARISON:")
            print("-" * 50)
            
            for alg in algorithms:
                if phase_data[alg]:
                    errors = [r['error'] for r in phase_data[alg]]
                    times = [r['time'] for r in phase_data[alg]]
                    evals = [r['evaluations'] for r in phase_data[alg]]
                    
                    print(f"\n{alg}:")
                    print(f"  Error: {np.mean(errors):.3f}% ± {np.std(errors):.3f}%")
                    print(f"  Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
                    print(f"  Evaluations: {np.mean(evals):.0f} ± {np.std(evals):.0f}")
                    
                    # Efficiency metrics
                    efficiency = np.mean(errors) / (np.mean(evals) / 1000)  # Error per 1000 evals
                    print(f"  Efficiency (error/1000 evals): {efficiency:.3f}")
                    
                    # Success rate
                    success_rate = np.sum(np.array(errors) < 5.0) / len(errors) * 100
                    print(f"  Success Rate (<5%): {success_rate:.1f}%")
                    
                    # Convergence analysis for Level 3
                    if self.comparison_level == 3:
                        conv_evals = [r['convergence_evaluation'] for r in phase_data[alg] 
                                      if r['convergence_evaluation'] > 0]
                        if conv_evals:
                            print(f"  Convergence: {len(conv_evals)}/{len(phase_data[alg])} runs")
                            print(f"  Mean convergence evals: {np.mean(conv_evals):.0f}")

# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

def run_fair_comparison_study():
    """Execute fair comparison study with both Level 1 and Level 3"""
    
    print("ENHANCED ADAPTIVE DIGITAL TWIN SYSTEM - FAIR COMPARISON STUDY")
    print("=" * 80)
    print("This study implements:")
    print("  1. Level 1: Fixed Budget Protocol (40k calibration, 12k adaptation)")
    print("  3. Level 3: Adaptive Convergence Protocol (stop on convergence)")
    print(f"  - Processing mode: Sequential (Windows-compatible, no multi-core)")
    print("  - Detailed evaluation tracking and efficiency metrics")
    print("=" * 80)
    
    # Motor parameters (2HP, 60Hz)
    ideal_motor_params = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
    
    # Run Level 1: Fixed Budget
    print("\n" + "="*80)
    print("EXECUTING LEVEL 1: FIXED BUDGET PROTOCOL")
    print("="*80)
    
    twin_system_L1 = EnhancedAdaptiveDigitalTwinSystem(
        ideal_motor_params, 
        comparison_level=1
    )
    results_L1 = twin_system_L1.run_enhanced_adaptive_study(n_runs=10)
    twin_system_L1.statistical_analysis_enhanced()
    
    # Run Level 3: Adaptive Convergence
    print("\n" + "="*80)
    print("EXECUTING LEVEL 3: ADAPTIVE CONVERGENCE PROTOCOL")
    print("="*80)
    
    twin_system_L3 = EnhancedAdaptiveDigitalTwinSystem(
        ideal_motor_params, 
        comparison_level=3
    )
    results_L3 = twin_system_L3.run_enhanced_adaptive_study(n_runs=10)
    twin_system_L3.statistical_analysis_enhanced()
    
    # Final comparison
    print(f"\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    print("\n📊 EFFICIENCY COMPARISON (Level 1 vs Level 3):")
    
    algorithms = list(results_L1.keys())
    
    for alg in algorithms:
        print(f"\n{alg}:")
        
        # Level 1 metrics
        L1_errors = results_L1[alg]['errors']
        L1_evals = results_L1[alg]['evaluations']
        L1_times = results_L1[alg]['times']
        
        # Level 3 metrics
        L3_errors = results_L3[alg]['errors']
        L3_evals = results_L3[alg]['evaluations']
        L3_times = results_L3[alg]['times']
        
        print(f"  Level 1 (Fixed Budget):")
        print(f"    - Mean error: {np.mean(L1_errors):.2f}%")
        print(f"    - Mean evaluations: {np.mean(L1_evals):.0f}")
        print(f"    - Mean time: {np.mean(L1_times):.2f}s")
        
        print(f"  Level 3 (Adaptive):")
        print(f"    - Mean error: {np.mean(L3_errors):.2f}%")
        print(f"    - Mean evaluations: {np.mean(L3_evals):.0f}")
        print(f"    - Mean time: {np.mean(L3_times):.2f}s")
        print(f"    - Evaluation savings: {(1 - np.mean(L3_evals)/np.mean(L1_evals))*100:.1f}%")
    
    print(f"\n" + "="*80)
    print("FAIR COMPARISON STUDY COMPLETED")
    print("="*80)
    print("\n📁 GENERATED FILES:")
    print("  Level 1 CSV Files:")
    for alg in algorithms:
        print(f"    • results/csv/{alg.replace(' ', '_')}_enhanced_fair_L1.csv")
    print("\n  Level 3 CSV Files:")
    for alg in algorithms:
        print(f"    • results/csv/{alg.replace(' ', '_')}_enhanced_fair_L3.csv")
    
    return twin_system_L1, twin_system_L3

if __name__ == "__main__":
    print("Starting Enhanced Fair Comparison Study...")
    print("This will run both Level 1 (Fixed Budget) and Level 3 (Adaptive Convergence)")
    print(f"Processing mode: Sequential (multi-core disabled)")
    print("-" * 80)
    
    study_results = run_fair_comparison_study()
    
    print(f"\n🎯 ENHANCED FAIR COMPARISON STUDY COMPLETED SUCCESSFULLY")
    print("All results have been saved to the 'results' directory.")
    print("The system provides scientifically rigorous algorithm comparison!")
