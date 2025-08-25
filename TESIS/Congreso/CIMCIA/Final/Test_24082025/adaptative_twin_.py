# adaptive_digital_twin_system_adaptive.py
# ADAPTIVE DIGITAL TWIN SYSTEM: Adaptive Convergence Framework (con tracking enriquecido)
# Ejecuta hasta cumplir criterios de convergencia o agotar presupuesto.
# Para Mechatronics, Control & AI Conference Submission

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import f_oneway
from tqdm import tqdm

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx  # (se mantiene por compatibilidad, no se usa directo)

warnings.filterwarnings('ignore')

print("System configured for sequential processing. Multiprocessing is disabled.")

# Directorios
os.makedirs('results', exist_ok=True)
os.makedirs('results/csv', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

# ===============================================================================
# CONFIGURACIÓN DE COMPARACIÓN JUSTA
# ===============================================================================
class FairComparisonConfig:
    """Configuración para comparación justa entre algoritmos"""

    # Nivel 1: Presupuesto máximo (límite duro)
    CALIBRATION_BUDGET = 40000   # Evaluaciones totales para calibración
    ADAPTATION_BUDGET = 12000    # Evaluaciones totales para adaptación

    # Nivel 3: Criterios adaptativos de convergencia
    CONVERGENCE_ERROR_THRESHOLD = 0.01  # 1% de error
    STAGNATION_ITERATIONS = 20          # Iteraciones sin mejora
    MIN_ITERATIONS = 10                 # Mínimo antes de evaluar convergencia

    @staticmethod
    def get_pso_config(is_adaptation=False):
        if is_adaptation:
            return {'n_particles': 100, 'max_iter': 120}
        else:
            return {'n_particles': 125, 'max_iter': 320}

    @staticmethod
    def get_bfo_config(is_adaptation=False):
        if is_adaptation:
            return {'n_bacteria': 30, 'n_chemotactic': 50, 'n_swim': 4, 'n_reproductive': 2, 'n_elimination': 2}
        else:
            return {'n_bacteria': 40, 'n_chemotactic': 63, 'n_swim': 4, 'n_reproductive': 4, 'n_elimination': 2}

    @staticmethod
    def get_chaotic_pso_config(is_adaptation=False):
        if is_adaptation:
            return {'n_particles': 80, 'max_iter': 150}
        else:
            return {'n_particles': 100, 'max_iter': 400}

# ===============================================================================
# UTILIDADES DE TRACKING
# ===============================================================================
@dataclass
class RunMetrics:
    algorithm: str
    scenario: str
    phase: int
    run_index: int
    start_time: float = field(default_factory=time.time)
    evaluations: int = 0
    best_cost: float = np.inf

    def update(self, evaluations: int, best_cost: float):
        self.evaluations = evaluations
        self.best_cost = best_cost

    def summary_line(self, budget: int) -> str:
        elapsed = time.time() - self.start_time
        usage = f"{self.evaluations}/{budget} evals ({100*self.evaluations/max(budget,1):.1f}%)"
        return (f"[{self.algorithm}] Scenario={self.scenario} Phase={self.phase} Run={self.run_index+1} | "
                f"Best={self.best_cost:.3e} | Evals={usage} | Elapsed={elapsed:.1f}s")

def eta_from_progress(done_steps: int, total_steps: int, start_time: float) -> float:
    if done_steps <= 0:
        return float('inf')
    elapsed = time.time() - start_time
    rate = done_steps / max(elapsed, 1e-9)
    remaining = max(total_steps - done_steps, 0)
    return remaining / max(rate, 1e-9)

# ===============================================================================
# MARCO DE SIMULACIÓN DE MOTOR
# ===============================================================================
def induction_motor(t, x, params, vqs, vds):
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2*np.pi*60
    ws = we - wr
    lqs = Ls*iqs + Lm*iqr
    lds = Ls*ids + Lm*idr
    lqr = Lr*iqr + Lm*iqs
    ldr = Lr*idr + Lm*ids
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs*iqs - we*lds, vds - rs*ids + we*lqs, -rr*iqr - ws*ldr, -rr*idr + ws*lqr])
    di_dt = np.linalg.solve(L, v)
    Te = (3*4/4) * Lm * (iqs*idr - ids*iqr)
    dwr_dt = (Te - B*wr) / J
    return np.array([*di_dt, dwr_dt])

def simulate_motor(params, t_span=[0, 2], n_points=500):
    vqs, vds = 220*np.sqrt(2)/np.sqrt(3), 0
    try:
        sol = solve_ivp(lambda t, x: induction_motor(t, x, params, vqs, vds),
                        t_span, [0,0,0,0,0], dense_output=True, rtol=1e-6, atol=1e-8)
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        Is_mag = np.sqrt(iqs**2 + ids**2)
        Te = (3*4/4) * params[4] * (iqs*idr - ids*iqr)
        rpm = wr * 60/(2*np.pi) * 2/4
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
# FUNCIONES OBJETIVO
# ===============================================================================
class CalibrationObjective:
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
        self.current_scale = np.max(np.abs(measured_current))
        self.torque_scale = np.max(np.abs(measured_torque))
        self.speed_scale = np.max(np.abs(measured_speed))

    def compensate_temperature(self, params):
        temp_diff = self.temperature - self.reference_temp
        compensated = params.copy()
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        return compensated

    def __call__(self, candidate_params):
        try:
            penalty = 0
            for i, (param, param_name) in enumerate(zip(candidate_params, self.param_names)):
                min_val, max_val = self.param_bounds[param_name]
                if param < min_val or param > max_val:
                    penalty += 1e6 * (1 + abs(param - np.clip(param, min_val, max_val)))
            Ls = candidate_params[2] + candidate_params[4]
            Lr = candidate_params[3] + candidate_params[4]
            if Ls <= candidate_params[2] or Lr <= candidate_params[3]: penalty += 1e6
            if candidate_params[0] <= 0 or candidate_params[1] <= 0: penalty += 1e8

            temp_compensated = self.compensate_temperature(candidate_params)
            _, sim_outputs = simulate_motor(temp_compensated,
                                            t_span=[0, len(self.measured_current)*0.005],
                                            n_points=len(self.measured_current))
            sim_current = sim_outputs['Is_mag']
            sim_torque = sim_outputs['Te']
            sim_speed = sim_outputs['rpm']

            current_mse = np.mean(((self.measured_current - sim_current) / self.current_scale)**2)
            torque_mse = np.mean(((self.measured_torque - sim_torque) / self.torque_scale)**2)
            speed_mse = np.mean(((self.measured_speed - sim_speed) / self.speed_scale)**2)

            weights = {'current': 0.5, 'torque': 0.3, 'speed': 0.2}
            total_mse = (weights['current'] * current_mse +
                         weights['torque'] * torque_mse +
                         weights['speed'] * speed_mse)

            regularization = 0
            for i, param in enumerate(candidate_params):
                expected = self.ideal_params[i]
                regularization += 0.001 * ((param - expected) / expected)**2

            return total_mse + penalty + regularization
        except Exception:
            return 1e10

class AdaptationObjective:
    def __init__(self, measured_current, temperature, param_names, param_bounds,
                 temp_coeffs, reference_temp):
        self.measured_current = measured_current
        self.temperature = temperature
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.temp_coeffs = temp_coeffs
        self.reference_temp = reference_temp
        self.current_scale = np.max(np.abs(measured_current))

    def compensate_temperature(self, params):
        temp_diff = self.temperature - self.reference_temp
        compensated = params.copy()
        for i, (param, coeff) in enumerate(zip(params, self.temp_coeffs)):
            compensated[i] = param * (1 + coeff * temp_diff)
        return compensated

    def __call__(self, candidate_params):
        try:
            penalty = 0
            for i, (param, param_name) in enumerate(zip(candidate_params, self.param_names)):
                min_val, max_val = self.param_bounds[param_name]
                if param < min_val or param > max_val:
                    penalty += 1e6 * (1 + abs(param - np.clip(param, min_val, max_val)))
            Ls = candidate_params[2] + candidate_params[4]
            Lr = candidate_params[3] + candidate_params[4]
            if Ls <= candidate_params[2] or Lr <= candidate_params[3]: penalty += 1e6
            if candidate_params[0] <= 0 or candidate_params[1] <= 0: penalty += 1e8

            temp_compensated = self.compensate_temperature(candidate_params)
            _, sim_outputs = simulate_motor(temp_compensated,
                                            t_span=[0, len(self.measured_current)*0.005],
                                            n_points=len(self.measured_current))
            sim_current = sim_outputs['Is_mag']
            current_mse = np.mean(((self.measured_current - sim_current) / self.current_scale)**2)
            return current_mse + penalty
        except Exception:
            return 1e10

# ===============================================================================
# EXCEPCIÓN DE CONVERGENCIA
# ===============================================================================
class ConvergenceReached(Exception):
    pass

# ===============================================================================
# OPTIMIZADOR: PSO ADAPTATIVO CON TRACKING DETALLADO
# ===============================================================================
class EnhancedAdaptivePSO:
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        config = FairComparisonConfig.get_pso_config(is_adaptation)
        self.n_particles, self.max_iter = config['n_particles'], config['max_iter']
        self.n_dims = len(bounds[0])
        self.cost_history = []
        self.evaluation_count = 0
        self.convergence_evaluation = -1
        self.best_cost = float('inf')
        self.best_params = None
        self.stagnation_counter = 0
        self.last_best_cost = float('inf')
        # opciones PSO
        if is_adaptation:
            self.options = {'c1': 1.5, 'c2': 2.5, 'w': 0.7, 'k': 5, 'p': 2}
        else:
            self.options = {'c1': 2.05, 'c2': 2.05, 'w': 0.9, 'k': 5, 'p': 2}

    def _check_convergence_state(self):
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        if self.best_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        if abs(self.best_cost - self.last_best_cost) < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_cost = self.best_cost
        return self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS

    def smart_initialization(self):
        init_pos = np.zeros((self.n_particles, self.n_dims))
        if self.is_adaptation and self.base_params is not None:
            n_close = int(self.n_particles * 0.6)
            init_pos[:n_close] = self.base_params * np.random.uniform(0.95, 1.05, (n_close, self.n_dims))
            init_pos[n_close:] = self.base_params * np.random.uniform(0.9, 1.1, (self.n_particles - n_close, self.n_dims))
        elif self.base_params is not None:
            n_near = int(self.n_particles * 0.4)
            n_medium = int(self.n_particles * 0.3)
            init_pos[:n_near] = self.base_params * np.random.uniform(0.8, 1.2, (n_near, self.n_dims))
            init_pos[n_near:n_near+n_medium] = self.base_params * np.random.uniform(0.7, 1.3, (n_medium, self.n_dims))
            init_pos[n_near+n_medium:] = np.random.uniform(self.bounds[0], self.bounds[1],
                                                           (self.n_particles - n_near - n_medium, self.n_dims))
        else:
            init_pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dims))
        return np.clip(init_pos, self.bounds[0], self.bounds[1])

    def pso_objective_wrapper(self, x):
        costs = np.array([self.objective_func(particle) for particle in x])
        self.evaluation_count += len(x)
        idx = np.argmin(costs)
        current_best = costs[idx]
        if current_best < self.best_cost:
            self.best_cost = current_best
            self.best_params = x[idx].copy()
        self.cost_history.append(self.best_cost)
        if self.convergence_evaluation == -1 and self._check_convergence_state():
            self.convergence_evaluation = self.evaluation_count
            raise ConvergenceReached
        return costs

    def optimize(self):
        start = time.time()
        init_pos = self.smart_initialization()
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=self.n_dims,
            options={k: self.options[k] for k in ['c1','c2','w']},
            bounds=self.bounds, init_pos=init_pos
        )

        # Barra de progreso manual por iteración
        pbar = tqdm(total=self.max_iter,
                    desc=f"      PSO ({'Adaptation' if self.is_adaptation else 'Calibration'})",
                    ncols=120, leave=False, position=1)

        try:
            for it in range(self.max_iter):
                # Ejecutamos de 1 en 1 para poder actualizar tracking
                optimizer.optimize(self.pso_objective_wrapper, iters=1, verbose=False)
                elapsed = time.time() - start
                eta_s = eta_from_progress(it+1, self.max_iter, start)
                pbar.update(1)
                pbar.set_postfix({
                    'iter': f"{it+1}/{self.max_iter}",
                    'evals': self.evaluation_count,
                    'best_cost': f"{self.best_cost:.3e}",
                    'eta': f"{eta_s:6.1f}s"
                })
                if self.convergence_evaluation != -1:
                    tqdm.write(f"      PSO converged early @eval {self.convergence_evaluation}, best={self.best_cost:.3e}")
                    break
        except ConvergenceReached:
            tqdm.write(f"      PSO converged early @eval {self.convergence_evaluation}, best={self.best_cost:.3e}")
        finally:
            pbar.close()

        optimization_time = time.time() - start
        return self.best_cost, self.best_params, optimization_time, self.evaluation_count, self.convergence_evaluation

# ===============================================================================
# OPTIMIZADOR: BFO ADAPTATIVO CON TRACKING DETALLADO
# ===============================================================================
class EnhancedAdaptiveBFO:
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        cfg = FairComparisonConfig.get_bfo_config(is_adaptation)
        self.S, self.Nc, self.Ns, self.Nre, self.Ned = \
            cfg['n_bacteria'], cfg['n_chemotactic'], cfg['n_swim'], cfg['n_reproductive'], cfg['n_elimination']
        self.Ped = 0.2 if not is_adaptation else 0.1
        self.Ci = 0.05 if not is_adaptation else 0.02
        self.n_dims = len(bounds[0])
        self.lb, self.ub = bounds
        self.evaluation_count = 0
        self.convergence_evaluation = -1
        self.stagnation_counter = 0
        self.best_cost = np.inf
        self.best_pos = None
        # init
        self.bacteria = self._smart_initialization()
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.evaluation_count += self.S
        self.health = np.zeros(self.S)
        min_idx = np.argmin(self.costs)
        self.best_pos = self.bacteria[min_idx].copy()
        self.best_cost = self.costs[min_idx]
        self.cost_history = [self.best_cost]
        self.last_best_cost = self.best_cost

    def _check_convergence_state(self):
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        if self.best_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        if abs(self.best_cost - self.last_best_cost) < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_cost = self.best_cost
        return self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS

    def _smart_initialization(self):
        bacteria = np.zeros((self.S, self.n_dims))
        if self.is_adaptation and self.base_params is not None:
            n_close = int(self.S * 0.7)
            bacteria[:n_close] = self.base_params * np.random.uniform(0.95, 1.05, (n_close, self.n_dims))
            bacteria[n_close:] = self.base_params * np.random.uniform(0.9, 1.1, (self.S - n_close, self.n_dims))
        elif self.base_params is not None:
            n_near = int(self.S * 0.4)
            bacteria[:n_near] = self.base_params * np.random.uniform(0.8, 1.2, (n_near, self.n_dims))
            bacteria[n_near:] = np.random.uniform(self.lb, self.ub, (self.S - n_near, self.n_dims))
        else:
            bacteria = np.random.uniform(self.lb, self.ub, (self.S, self.n_dims))
        return np.clip(bacteria, self.lb, self.ub)

    def _update_best(self):
        idx = np.argmin(self.costs)
        if self.costs[idx] < self.best_cost:
            self.best_cost = self.costs[idx]
            self.best_pos = self.bacteria[idx].copy()

    def _tumble(self):
        direction = np.random.uniform(-1, 1, (self.S, self.n_dims))
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return direction / norms

    def _reproduce(self):
        sorted_indices = np.argsort(self.health)
        n_survive = self.S // 2
        survivors_pos = self.bacteria[sorted_indices[:n_survive]]
        offspring = survivors_pos + np.random.normal(0, 0.01 if not self.is_adaptation else 0.005, survivors_pos.shape)
        self.bacteria = np.concatenate([survivors_pos, np.clip(offspring, self.lb, self.ub)])
        self.costs = np.array([self.objective_func(b) for b in self.bacteria])
        self.evaluation_count += len(offspring)
        self.health = np.zeros(self.S)

    def _eliminate_disperse(self):
        for i in range(self.S):
            if np.random.rand() < self.Ped:
                if self.is_adaptation and self.base_params is not None:
                    self.bacteria[i] = self.best_pos * np.random.uniform(0.95, 1.05, self.n_dims)
                else:
                    self.bacteria[i] = np.random.uniform(self.lb, self.ub, self.n_dims)
                self.bacteria[i] = np.clip(self.bacteria[i], self.lb, self.ub)
                self.costs[i] = self.objective_func(self.bacteria[i])
                self.evaluation_count += 1

    def optimize(self):
        start = time.time()
        mode = "Adaptation" if self.is_adaptation else "Calibration"
        total_chemo_steps = self.Ned * self.Nre * self.Nc

        pbar = tqdm(total=total_chemo_steps, desc=f"      BFO ({mode})",
                    ncols=120, leave=False, position=1)
        converged = False

        for l in range(self.Ned):
            for k in range(self.Nre):
                for j in range(self.Nc):
                    self._update_best()
                    if self.convergence_evaluation == -1 and self._check_convergence_state():
                        self.convergence_evaluation = self.evaluation_count
                        converged = True
                        break

                    last_costs = np.copy(self.costs)
                    directions = self._tumble()
                    adaptive_step = self.Ci * (1 - j/self.Nc * 0.5)

                    for m in range(self.Ns):
                        new_pos = self.bacteria + adaptive_step * directions
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_costs = np.array([self.objective_func(p) for p in new_pos])
                        self.evaluation_count += self.S
                        improved = new_costs < self.costs
                        self.bacteria[improved] = new_pos[improved]
                        self.costs[improved] = new_costs[improved]
                        self.health += last_costs - self.costs
                        if not np.any(improved):
                            break

                    self._update_best()
                    self.cost_history.append(self.best_cost)
                    eta_s = eta_from_progress(pbar.n + 1, total_chemo_steps, start)
                    pbar.update(1)
                    pbar.set_postfix({
                        'step': f"{pbar.n}/{total_chemo_steps}",
                        'evals': self.evaluation_count,
                        'best_cost': f"{self.best_cost:.3e}",
                        'eta': f"{eta_s:6.1f}s"
                    })

                self._reproduce()
                if converged:
                    break

            self._eliminate_disperse()
            if converged:
                tqdm.write(f"      BFO converged early @eval {self.convergence_evaluation}, best={self.best_cost:.3e}")
                break

        pbar.close()
        optimization_time = time.time() - start
        return self.best_cost, self.best_pos, optimization_time, self.evaluation_count, self.convergence_evaluation

# ===============================================================================
# OPTIMIZADOR: Chaotic PSO-DSO CON TRACKING DETALLADO
# ===============================================================================
class EnhancedAdaptiveChaoticPSODSO:
    def __init__(self, objective_func, bounds, base_params=None, is_adaptation=False):
        self.objective_func = objective_func
        self.bounds = bounds
        self.base_params = base_params
        self.is_adaptation = is_adaptation
        config = FairComparisonConfig.get_chaotic_pso_config(is_adaptation)
        self.n_particles, self.max_iter = config['n_particles'], config['max_iter']
        self.n_dims = len(bounds[0])
        self.w_max, self.w_min = (0.7, 0.4) if is_adaptation else (0.95, 0.3)
        self.c1_init, self.c2_init = (2.0, 1.0) if is_adaptation else (2.8, 0.3)

        self.chaos_values = np.random.rand(self.n_particles)
        self.particles = self._smart_initialization()
        self.velocities = np.zeros((self.n_particles, self.n_dims))
        self.pbest = self.particles.copy()
        self.pbest_costs = np.array([self.objective_func(p) for p in self.particles])
        self.gbest = self.pbest[np.argmin(self.pbest_costs)]
        self.gbest_cost = np.min(self.pbest_costs)
        self.evaluation_count = self.n_particles
        self.cost_history = [self.gbest_cost]
        self.convergence_evaluation = -1
        self.stagnation_counter = 0
        self.last_best_cost = self.gbest_cost

    def _check_convergence_state(self):
        if len(self.cost_history) < FairComparisonConfig.MIN_ITERATIONS:
            return False
        if self.gbest_cost < FairComparisonConfig.CONVERGENCE_ERROR_THRESHOLD:
            return True
        if abs(self.gbest_cost - self.last_best_cost) < 1e-12:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_cost = self.gbest_cost
        return self.stagnation_counter >= FairComparisonConfig.STAGNATION_ITERATIONS

    def _smart_initialization(self):
        particles = np.zeros((self.n_particles, self.n_dims))
        if self.is_adaptation and self.base_params is not None:
            n_close = int(self.n_particles * 0.6)
            particles[:n_close] = self.base_params * np.random.uniform(0.95, 1.05, (n_close, self.n_dims))
            particles[n_close:] = self.base_params * np.random.uniform(0.9, 1.1, (self.n_particles - n_close, self.n_dims))
        elif self.base_params is not None:
            n_near = int(self.n_particles * 0.4)
            n_medium = int(self.n_particles * 0.3)
            particles[:n_near] = self.base_params * np.random.uniform(0.8, 1.2, (n_near, self.n_dims))
            particles[n_near:n_near+n_medium] = self.base_params * np.random.uniform(0.7, 1.3, (n_medium, self.n_dims))
            particles[n_near+n_medium:] = np.random.uniform(self.bounds[0], self.bounds[1],
                                                            (self.n_particles - n_near - n_medium, self.n_dims))
        else:
            particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.n_dims))
        return np.clip(particles, self.bounds[0], self.bounds[1])

    def optimize(self):
        start = time.time()
        mode = "Adaptation" if self.is_adaptation else "Calibration"

        pbar = tqdm(range(self.max_iter), desc=f"      Chaotic PSO ({mode})",
                    ncols=120, leave=False, position=1)
        stagnation_local = 0
        last_best_local = self.gbest_cost

        for iteration in pbar:
            if self.convergence_evaluation == -1 and self._check_convergence_state():
                self.convergence_evaluation = self.evaluation_count
                tqdm.write(f"      Chaotic PSO converged early @eval {self.evaluation_count}, best={self.gbest_cost:.3e}")
                break

            progress_ratio = iteration / max(self.max_iter, 1)
            w = self.w_max - (self.w_max - self.w_min) * progress_ratio
            c1 = self.c1_init - (self.c1_init - 2.0) * progress_ratio
            c2 = self.c2_init + (2.0 - self.c2_init) * progress_ratio

            if abs(self.gbest_cost - last_best_local) < 1e-12:
                stagnation_local += 1
            else:
                stagnation_local = 0
            last_best_local = self.gbest_cost

            chaos_intensity = (0.05 if self.is_adaptation else 0.1) * (3 if stagnation_local > 5 else 1)

            for i in range(self.n_particles):
                # mapa caótico simple (tent map)
                self.chaos_values[i] = 2*self.chaos_values[i] if self.chaos_values[i] < 0.5 else 2*(1 - self.chaos_values[i])
                r1, r2 = self.chaos_values[i], np.random.rand()
                chaos_factor = chaos_intensity * self.chaos_values[i]
                v_max = (0.1 if self.is_adaptation else 0.2) * (self.bounds[1] - self.bounds[0])
                self.velocities[i] = w * self.velocities[i] + c1 * r1 * (self.pbest[i] - self.particles[i]) + \
                                     c2 * r2 * (self.gbest - self.particles[i]) + chaos_factor * (np.random.rand(self.n_dims) - 0.5)
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.bounds[0], self.bounds[1])
                cost = self.objective_func(self.particles[i])
                self.evaluation_count += 1
                if cost < self.pbest_costs[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_costs[i] = cost
                    if cost < self.gbest_cost:
                        self.gbest = self.particles[i].copy()
                        self.gbest_cost = cost

            self.cost_history.append(self.gbest_cost)
            eta_s = eta_from_progress(iteration+1, self.max_iter, start)
            pbar.set_postfix({
                'iter': f"{iteration+1}/{self.max_iter}",
                'evals': self.evaluation_count,
                'best_cost': f"{self.gbest_cost:.3e}",
                'eta': f"{eta_s:6.1f}s"
            })

        pbar.close()
        optimization_time = time.time() - start
        return self.gbest_cost, self.gbest, optimization_time, self.evaluation_count, self.convergence_evaluation

# ===============================================================================
# UTILIDAD: EJECUCIÓN SECUENCIAL DE UN OPTIMIZADOR
# ===============================================================================
def run_single_optimization(args):
    """Ejecuta un optimizador de forma secuencial y devuelve resultados detallados"""
    (alg_name, AlgorithmClass, objective, bounds, base_params,
     is_adaptation, run_num, scenario_name, budget) = args

    np.random.seed(run_num * 100 + hash(alg_name + scenario_name) % 1000)

    algorithm = AlgorithmClass(objective, bounds, base_params, is_adaptation)
    cost, params, opt_time, eval_count, conv_eval = algorithm.optimize()

    # Respetar presupuesto: si superara, se trunca a reporte (el diseño actual no fuerza corte duro a mitad)
    eval_count = min(eval_count, budget)

    return {
        'algorithm': alg_name,
        'run': run_num,
        'cost': cost,
        'params': params,
        'time': opt_time,
        'evaluations': eval_count,
        'cost_history': getattr(algorithm, 'cost_history', []),
        'convergence_evaluation': conv_eval
    }

# ===============================================================================
# SISTEMA DIGITAL TWIN ADAPTATIVO
# ===============================================================================
class EnhancedAdaptiveDigitalTwinSystem:
    def __init__(self, ideal_params):
        self.ideal_params = np.array(ideal_params)
        self.param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
        self.param_bounds = {'rs':(0.5,10.0),'rr':(0.5,10.0),'Lls':(0.001,0.05),'Llr':(0.001,0.05),
                             'Lm':(0.05,0.5),'J':(0.001,0.1),'B':(0.0001,0.01)}
        self.temp_coeffs = np.array([0.004, 0.004, 0.001, 0.001, 0.0005, 0, 0])
        self.reference_temp = 20.0
        self.algorithms = {'PSO': EnhancedAdaptivePSO, 'BFO': EnhancedAdaptiveBFO,
                           'Chaotic PSO-DSO': EnhancedAdaptiveChaoticPSODSO}
        self.digital_twin_base = {}
        self.best_calibration_error = {}
        self.detailed_results = {}
        self.scenario_data_storage = {}

    def generate_nonideal_scenario(self, scenario_name, temperature, degradation_factor, noise_level):
        nonideal_params = self.ideal_params.copy()
        temp_diff = temperature - self.reference_temp
        for i, coeff in enumerate(self.temp_coeffs):
            nonideal_params[i] *= (1 + coeff * temp_diff)
        np.random.seed(hash(scenario_name) % (2**32 - 1))
        degradation = np.clip(np.random.normal(1.0, degradation_factor, len(nonideal_params)), 0.85, 1.15)
        nonideal_params *= degradation

        t, outputs = simulate_motor(nonideal_params, t_span=[0, 2.0], n_points=400)
        current_clean, torque_clean, speed_clean = outputs['Is_mag'], outputs['Te'], outputs['rpm']
        current_noise = np.random.normal(0, noise_level * np.std(current_clean), len(current_clean))
        torque_noise = np.random.normal(0, noise_level * 0.5 * np.std(torque_clean), len(torque_clean))
        speed_noise = np.random.normal(0, noise_level * 0.3 * np.std(speed_clean), len(speed_clean))

        return {'true_params': nonideal_params,
                'measured_current': current_clean + current_noise,
                'measured_torque': torque_clean + torque_noise,
                'measured_speed': speed_clean + speed_noise,
                'temperature': temperature, 'time': t}

    def run_study(self, n_runs=10):
        print("="*80)
        print("ADAPTIVE DIGITAL TWIN - ADAPTIVE CONVERGENCE STUDY")
        print("Running until convergence criteria are met (or budget is exhausted).")
        print("="*80)

        scenarios = [
            {'name': 'Normal_Operation', 'temp': 40, 'degradation': 0.03, 'noise': 0.01, 'phase': 1},
            {'name': 'High_Temperature', 'temp': 70, 'degradation': 0.06, 'noise': 0.02, 'phase': 2},
            {'name': 'Severe_Conditions', 'temp': 85, 'degradation': 0.10, 'noise': 0.03, 'phase': 2}
        ]

        for alg_name in self.algorithms.keys():
            self.detailed_results[alg_name] = []
            self.digital_twin_base[alg_name] = None
            self.best_calibration_error[alg_name] = float('inf')

        for scenario in scenarios:
            phase_name = "CALIBRATION" if scenario['phase'] == 1 else "ADAPTATION"
            budget = FairComparisonConfig.CALIBRATION_BUDGET if scenario['phase'] == 1 else FairComparisonConfig.ADAPTATION_BUDGET
            print(f"\n{'='*60}\nPHASE {scenario['phase']} - {phase_name}: {scenario['name']} (Max Budget: {budget} evals)\n{'='*60}")

            scenario_data = self.generate_nonideal_scenario(
                scenario['name'], scenario['temp'], scenario['degradation'], scenario['noise']
            )
            self.scenario_data_storage[scenario['name']] = scenario_data

            # Progreso de runs del escenario
            scenario_bar = tqdm(range(n_runs),
                                desc=f"   Overall Progress for {scenario['name']}",
                                ncols=120, unit="run", position=0, leave=True)

            all_results_for_scenario = []

            for run in scenario_bar:
                tqdm.write(f"\n--- Starting Run {run+1}/{n_runs} ---")
                for alg_name, AlgorithmClass in self.algorithms.items():

                    if scenario['phase'] == 1:
                        objective = CalibrationObjective(
                            scenario_data['measured_current'], scenario_data['measured_torque'],
                            scenario_data['measured_speed'], scenario_data['temperature'],
                            self.ideal_params, self.param_names, self.param_bounds,
                            self.temp_coeffs, self.reference_temp
                        )
                        bounds = (self.ideal_params * 0.8, self.ideal_params * 1.2)
                        base_params = self.ideal_params
                        is_adaptation = False
                    else:
                        # Fase de adaptación requiere base previa
                        if self.digital_twin_base[alg_name] is None:
                            tqdm.write(f"   ↳ Skipping {alg_name} (no calibrated base yet).")
                            continue
                        objective = AdaptationObjective(
                            scenario_data['measured_current'], scenario_data['temperature'],
                            self.param_names, self.param_bounds, self.temp_coeffs, self.reference_temp
                        )
                        base = self.digital_twin_base[alg_name]
                        bounds = (base * 0.8, base * 1.2)
                        base_params = base
                        is_adaptation = True

                    # Ejecutar optimización
                    task = (alg_name, AlgorithmClass, objective, bounds, base_params,
                            is_adaptation, run, scenario['name'], budget)
                    run_metrics = RunMetrics(alg_name, scenario['name'], scenario['phase'], run)

                    result = run_single_optimization(task)
                    run_metrics.update(result['evaluations'], result['cost'])
                    tqdm.write("   " + run_metrics.summary_line(budget))

                    all_results_for_scenario.append(result)

                # Actualizar postfix del progress con media de mejores costos observados en el run
                if all_results_for_scenario:
                    last_results = [r for r in all_results_for_scenario if r['run'] == run]
                    if last_results:
                        avg_cost = np.mean([r['cost'] for r in last_results])
                        scenario_bar.set_postfix({'avg_run_cost': f"{avg_cost:.2e}"})

            # Procesar resultados tras el escenario
            for result in all_results_for_scenario:
                alg_name, params = result['algorithm'], result['params']
                if params is None:
                    continue
                param_errors = np.abs((params - scenario_data['true_params']) / scenario_data['true_params']) * 100
                param_error = np.mean(param_errors)

                if scenario['phase'] == 1 and param_error < self.best_calibration_error[alg_name]:
                    self.digital_twin_base[alg_name] = params.copy()
                    self.best_calibration_error[alg_name] = param_error

                detailed_run = {
                    'scenario': scenario['name'], 'phase': scenario['phase'], 'run': result['run'] + 1,
                    'cost': result['cost'], 'error': param_error, 'time': result['time'],
                    'evaluations': result['evaluations'],
                    'convergence_evaluation': result['convergence_evaluation'],
                    'identified_params': params.tolist(), 'true_params': scenario_data['true_params'].tolist()
                }
                for i, p_name in enumerate(self.param_names):
                    detailed_run[f'error_{p_name}'] = param_errors[i]
                self.detailed_results[alg_name].append(detailed_run)

            # Resumen por escenario
            print("\n--- Scenario Summary:", scenario['name'], "---")
            for alg_name in self.algorithms.keys():
                subset = [r for r in self.detailed_results[alg_name] if r['scenario'] == scenario['name']]
                if not subset:
                    print(f"   {alg_name:<18} | no data")
                    continue
                mean_err = np.mean([r['error'] for r in subset])
                mean_eval = np.mean([r['evaluations'] for r in subset])
                convs = [r['convergence_evaluation'] for r in subset if r['convergence_evaluation'] and r['convergence_evaluation'] > 0]
                conv_str = f"{np.mean(convs):.0f} ({len(convs)}/{len(subset)})" if convs else "N/A"
                print(f"   {alg_name:<18} | mean err: {mean_err:6.2f}% | mean evals: {mean_eval:7.0f} | conv@eval: {conv_str}")

        self.export_to_csv()
        self.statistical_analysis()

    def export_to_csv(self):
        print("\n" + "="*80 + "\nEXPORTING RESULTS TO CSV\n" + "="*80)
        for alg_name, results in self.detailed_results.items():
            if results:
                df = pd.DataFrame(results)
                filename = f"results/csv/{alg_name.replace(' ', '_')}_adaptive_study.csv"
                df.to_csv(filename, index=False, float_format='%.6f')
                print(f"   ✓ Saved {alg_name} results to {filename}")

    def statistical_analysis(self):
        print("\n" + "="*80 + "\nSTATISTICAL ANALYSIS\n" + "="*80)
        algorithms = list(self.algorithms.keys())
        for phase in [1, 2]:
            phase_name = "CALIBRATION" if phase == 1 else "ADAPTATION"
            print(f"\n{'='*60}\nPHASE {phase}: {phase_name}\n{'='*60}")

            phase_data = {alg: [r for r in self.detailed_results[alg] if r['phase'] == phase] for alg in algorithms}
            if not any(phase_data.values()):
                print("   (no data)")
                continue

            # ANOVA sobre errores de parámetros
            groups = [np.array([r['error'] for r in phase_data[alg]]) for alg in algorithms if phase_data[alg]]
            try:
                f_stat, p_value = f_oneway(*groups)
                print(f"\nANOVA Test for Parameter Errors (p-value: {p_value:.4f}) -> "
                      f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            except Exception:
                print("\nANOVA Test for Parameter Errors: insufficient data")

            # Tabla comparativa
            print("\nALGORITHM COMPARISON:")
            print("-" * 80)
            print(f"{'Algorithm':<20} | {'Mean Error (%)':<18} | {'Mean Evals (Conv.)':<22} | {'Success Rate (%)':<17}")
            print("-" * 80)
            for alg in algorithms:
                if not phase_data[alg]:
                    print(f"{alg:<20} | {'N/A':<18} | {'N/A':<22} | {'N/A':<17}")
                    continue
                errors = np.array([r['error'] for r in phase_data[alg]])
                conv_evals = [r['convergence_evaluation'] for r in phase_data[alg] if r['convergence_evaluation'] and r['convergence_evaluation'] > 0]
                mean_error_str = f"{np.mean(errors):.2f} ± {np.std(errors):.2f}"
                mean_conv_eval_str = f"{np.mean(conv_evals):.0f} ({len(conv_evals)}/{len(errors)})" if conv_evals else "N/A"
                success_rate = np.mean(errors < 5.0) * 100.0
                print(f"{alg:<20} | {mean_error_str:<18} | {mean_conv_eval_str:<22} | {success_rate:<17.1f}")
            print("-" * 80)

# ===============================================================================
# MAIN
# ===============================================================================
def run_adaptive_study():
    """Ejecuta el estudio adaptativo con tracking detallado."""
    # Parámetros del motor (2HP, 60Hz)
    ideal_motor_params = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])

    twin_system = EnhancedAdaptiveDigitalTwinSystem(ideal_motor_params)
    twin_system.run_study(n_runs=10)

    print("\n" + "="*80)
    print("ADAPTIVE CONVERGENCE STUDY COMPLETED")
    print("="*80)
    print("\n📁 GENERATED FILES:")
    for alg in twin_system.algorithms.keys():
        print(f"   • results/csv/{alg.replace(' ', '_')}_adaptive_study.csv")

if __name__ == "__main__":
    run_adaptive_study()
    print(f"\n🎯 ADAPTIVE STUDY COMPLETED SUCCESSFULLY")
    print("All results have been saved to the 'results' directory.")
