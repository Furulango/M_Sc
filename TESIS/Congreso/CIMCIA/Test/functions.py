import numpy as np
import pyswarms as ps
from scipy.optimize import minimize

def run_pso(objective_func, bounds, n_particles, iterations, options={'c1': 2.05, 'c2': 2.05, 'w': 0.9}):
    """
    Ejecuta el algoritmo Particle Swarm Optimization (PSO).

    Args:
        objective_func (function): La función a minimizar. Debe aceptar un vector de NumPy.
        bounds (tuple): Una tupla de (límites_inferiores, límites_superiores) como arrays de NumPy.
        n_particles (int): El número de partículas en el enjambre.
        iterations (int): El número de iteraciones a ejecutar.
        options (dict): Diccionario de hiperparámetros de PSO (c1, c2, w).

    Returns:
        tuple: (mejor_costo, mejor_posición)
    """
    # Función de envoltura, pyswarms pueda evaluar el enjambre
    def pso_wrapper(x):
        return np.array([objective_func(p) for p in x])

    # Creación del optimizador
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options=options,
        bounds=bounds
    )

    # Ejecución de la optimización
    best_cost, best_pos = optimizer.optimize(pso_wrapper, iters=iterations, verbose=False)

    return float(best_cost), best_pos

def run_pso_sqp(objective_func, bounds, n_particles, pso_iterations, pso_options={'c1': 2.05, 'c2': 2.05, 'w': 0.9}):
    """
    Ejecuta un algoritmo híbrido PSO-SQP.

    Args:
        objective_func (function): La función a minimizar.
        bounds (tuple): Una tupla de (límites_inferiores, límites_superiores).
        n_particles (int): Número de partículas para la fase de PSO.
        pso_iterations (int): Número de iteraciones para la fase de PSO.
        pso_options (dict): Opciones para PSO.

    Returns:
        tuple: (mejor_costo, mejor_posición)
    """
    print("--- Fase 1: Búsqueda Global con PSO ---")
    pso_cost, pso_pos = run_pso(
        objective_func,
        bounds,
        n_particles,
        pso_iterations,
        pso_options
    )
    
    print(f"Resultado de PSO: Costo={pso_cost:.6e}")
    print("--- Fase 2: Refinamiento Local con SQP ---")
    
    # Prepara los límites para scipy.optimize.minimize
    scipy_bounds = list(zip(bounds[0], bounds[1]))

    sqp_result = minimize(
        objective_func,
        pso_pos, # Inicia desde la mejor posición de PSO
        method='SLSQP',
        bounds=scipy_bounds,
        options={'ftol': 1e-9, 'maxiter': 200, 'disp': False}
    )

    # Compara el resultado de SQP con el de PSO
    if sqp_result.success and sqp_result.fun < pso_cost:
        print(f"SQP mejoró el resultado a: Costo={sqp_result.fun:.6e}")
        return float(sqp_result.fun), sqp_result.x
    else:
        print("SQP no mejoró el resultado. Se mantiene la solución de PSO.")
        return pso_cost, pso_pos

class BacterialForaging:
    """
    Implementación del Algoritmo de Búsqueda por Forrajeo Bacteriano (BFO).
    """
    def __init__(self, objective_func, bounds, n_bacteria=20, n_chemotactic=30, n_swim=4,
                 n_reproductive=4, n_elimination=2, p_eliminate=0.25, step_size=0.1):
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

    def optimize(self):
        for l in range(self.Ned):
            for k in range(self.Nre):
                # Ciclo chemotáctico
                for j in range(self.Nc):
                    self._update_best()
                    
                    last_costs = np.copy(self.costs)
                    directions = self._tumble()
                    
                    # Natación (Swim)
                    for m in range(self.Ns):
                        new_pos = self.bacteria + self.Ci * directions
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_costs = np.array([self.objective_func(p) for p in new_pos])
                        
                        improved_mask = new_costs < self.costs
                        self.bacteria[improved_mask] = new_pos[improved_mask]
                        self.costs[improved_mask] = new_costs[improved_mask]
                        self.health += last_costs - self.costs
                        
                        if not np.any(improved_mask):
                            break # Si ninguna bacteria mejora, termina la natación

                # Reproducción
                self._reproduce()

            # Eliminación y dispersión
            self._eliminate_disperse()
            
        self._update_best()
        return self.best_cost, self.best_pos

    def _update_best(self):
        min_cost_idx = np.argmin(self.costs)
        if self.costs[min_cost_idx] < self.best_cost:
            self.best_cost = self.costs[min_cost_idx]
            self.best_pos = self.bacteria[min_cost_idx]

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