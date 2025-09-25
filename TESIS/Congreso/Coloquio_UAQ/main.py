import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

os.makedirs('results', exist_ok=True)

N_RUNS = 3
TRUE_PARAMS = np.array([2.45, 1.83, 0.008, 0.008, 0.203, 0.02, 0.001])
PARAM_NAMES = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
PARAM_BOUNDS = (TRUE_PARAMS * 0.8, TRUE_PARAMS * 1.2)  # Rango más pequeño

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
    
    di_dt = np.linalg.solve(L_matrix, v_vector)
    Te = (3 * 4 / 4) * Lm * (iqs * idr - ids * iqr)
    dwr_dt = (Te - B * wr) / J
    
    return np.array([*di_dt, dwr_dt])

def simulate_motor_simple(params):
    """Versión súper simplificada de la simulación"""
    vqs = 220 * np.sqrt(2) / np.sqrt(3)
    vds = 0
    initial_state = [0, 0, 0, 0, 0]
    
    # Solo 50 puntos en 0.1 segundos
    t_span = [0, 0.1]
    n_points = 50
    
    try:
        sol = solve_ivp(
            fun=lambda t, x: induction_motor_model(t, x, params, vqs, vds),
            t_span=t_span,
            y0=initial_state,
            method='RK23',  # Método más rápido
            t_eval=np.linspace(t_span[0], t_span[1], n_points),
            rtol=1e-3,  # Tolerancia muy relajada
            atol=1e-5
        )
        
        iqs, ids = sol.y[0], sol.y[1]
        wr = sol.y[4]
        
        Is_mag = np.sqrt(iqs**2 + ids**2)
        rpm = wr * 60 / (2 * np.pi)
        
        return Is_mag, rpm
    except:
        return np.ones(n_points) * 1000, np.zeros(n_points)

class SimpleObjective:
    def __init__(self, target_current, target_rpm):
        self.target_current = target_current
        self.target_rpm = target_rpm
        self.eval_count = 0

    def __call__(self, params):
        self.eval_count += 1
        
        if any(p <= 0 for p in params):
            return 1e6
        
        sim_current, sim_rpm = simulate_motor_simple(params)
        
        current_error = np.mean((self.target_current - sim_current)**2)
        rpm_error = np.mean((self.target_rpm - sim_rpm)**2)
        
        return current_error + rpm_error * 0.001  # RPM con menos peso

class SimplePSO:
    def __init__(self, objective_func, bounds, n_particles=20, max_iter=50):
        self.obj = objective_func
        self.lb, self.ub = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.n_dims = len(self.lb)

    def optimize(self):
        # Inicialización
        X = np.random.uniform(self.lb, self.ub, (self.n_particles, self.n_dims))
        V = np.zeros((self.n_particles, self.n_dims))
        pbest = X.copy()
        pbest_cost = np.array([self.obj(x) for x in X])
        gbest_idx = np.argmin(pbest_cost)
        gbest = pbest[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]
        
        print(f"    Iteración inicial - mejor costo: {gbest_cost:.6f}")
        
        for it in range(self.max_iter):
            # Actualizar velocidades y posiciones
            r1, r2 = np.random.rand(2)
            V = 0.7 * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
            X = X + V
            X = np.clip(X, self.lb, self.ub)
            
            # Evaluar
            for i in range(self.n_particles):
                cost = self.obj(X[i])
                if cost < pbest_cost[i]:
                    pbest[i] = X[i]
                    pbest_cost[i] = cost
                    if cost < gbest_cost:
                        gbest = X[i].copy()
                        gbest_cost = cost
            
            if it % 10 == 0:
                print(f"    Iteración {it}/{self.max_iter} - mejor costo: {gbest_cost:.6f}")
        
        return gbest_cost, gbest

class SimpleGWO:
    def __init__(self, objective_func, bounds, n_wolves=20, max_iter=50):
        self.obj = objective_func
        self.lb, self.ub = bounds
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.n_dims = len(self.lb)

    def optimize(self):
        # Inicialización
        X = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.n_dims))
        
        # Evaluar población inicial
        fitness = np.array([self.obj(x) for x in X])
        sorted_idx = np.argsort(fitness)
        
        alpha_pos = X[sorted_idx[0]].copy()
        beta_pos = X[sorted_idx[1]].copy()
        delta_pos = X[sorted_idx[2]].copy()
        
        print(f"    Iteración inicial - mejor costo: {fitness[sorted_idx[0]]:.6f}")
        
        for it in range(self.max_iter):
            a = 2 - it * (2 / self.max_iter)
            
            for i in range(self.n_wolves):
                # Actualizar posición basada en alpha, beta, delta
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                X1 = alpha_pos - A1 * abs(C1 * alpha_pos - X[i])
                
                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                X2 = beta_pos - A2 * abs(C2 * beta_pos - X[i])
                
                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                X3 = delta_pos - A3 * abs(C3 * delta_pos - X[i])
                
                X[i] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], self.lb, self.ub)
            
            # Re-evaluar y actualizar líderes
            fitness = np.array([self.obj(x) for x in X])
            sorted_idx = np.argsort(fitness)
            
            alpha_pos = X[sorted_idx[0]].copy()
            beta_pos = X[sorted_idx[1]].copy()
            delta_pos = X[sorted_idx[2]].copy()
            
            if it % 10 == 0:
                print(f"    Iteración {it}/{self.max_iter} - mejor costo: {fitness[sorted_idx[0]]:.6f}")
        
        return fitness[sorted_idx[0]], alpha_pos

def main():
    print("="*60)
    print("COMPARACIÓN PSO vs GWO - Versión Simplificada")
    print("="*60)
    
    # Generar datos objetivo
    print("\n1. Generando datos de referencia...")
    target_current, target_rpm = simulate_motor_simple(TRUE_PARAMS)
    print(f"   Datos generados: {len(target_current)} puntos")
    
    results = []
    
    # PSO
    print("\n2. Ejecutando PSO...")
    for run in range(N_RUNS):
        print(f"\n   Run {run+1}/{N_RUNS}:")
        start = time.time()
        obj = SimpleObjective(target_current, target_rpm)
        pso = SimplePSO(obj, PARAM_BOUNDS)
        best_cost, best_params = pso.optimize()
        elapsed = time.time() - start
        
        param_error = np.mean(np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)) * 100
        
        results.append({
            'algorithm': 'PSO',
            'run': run + 1,
            'cost': best_cost,
            'param_error_%': param_error,
            'time_s': elapsed
        })
        print(f"   Completado en {elapsed:.2f}s - Error: {param_error:.2f}%")
    
    # GWO
    print("\n3. Ejecutando GWO...")
    for run in range(N_RUNS):
        print(f"\n   Run {run+1}/{N_RUNS}:")
        start = time.time()
        obj = SimpleObjective(target_current, target_rpm)
        gwo = SimpleGWO(obj, PARAM_BOUNDS)
        best_cost, best_params = gwo.optimize()
        elapsed = time.time() - start
        
        param_error = np.mean(np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)) * 100
        
        results.append({
            'algorithm': 'GWO',
            'run': run + 1,
            'cost': best_cost,
            'param_error_%': param_error,
            'time_s': elapsed
        })
        print(f"   Completado en {elapsed:.2f}s - Error: {param_error:.2f}%")
    
    # Guardar y mostrar resultados
    df = pd.DataFrame(results)
    df.to_csv('results/comparison_simple.csv', index=False)
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    
    for alg in ['PSO', 'GWO']:
        alg_data = df[df['algorithm'] == alg]
        print(f"\n{alg}:")
        print(f"  Error promedio: {alg_data['param_error_%'].mean():.2f}%")
        print(f"  Tiempo promedio: {alg_data['time_s'].mean():.2f}s")
        print(f"  Mejor resultado: {alg_data['param_error_%'].min():.2f}%")
    
    print("\n" + "="*60)
    pso_err = df[df['algorithm'] == 'PSO']['param_error_%'].mean()
    gwo_err = df[df['algorithm'] == 'GWO']['param_error_%'].mean()
    
    if pso_err < gwo_err:
        print(f"GANADOR: PSO (mejor por {gwo_err - pso_err:.2f}%)")
    else:
        print(f"GANADOR: GWO (mejor por {pso_err - gwo_err:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()