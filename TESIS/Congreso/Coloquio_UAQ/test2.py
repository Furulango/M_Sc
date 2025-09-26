import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

# reproducibilidad (opcional)
np.random.seed(0)

N_RUNS = 3
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
    ws = we - wr

    lqs = Ls * iqs + Lm * iqr
    lds = Ls * ids + Lm * idr
    lqr = Lr * iqr + Lm * iqs
    ldr = Lr * idr + Lm * ids

    L_matrix = np.array([[Ls, 0, Lm, 0],
                         [0, Ls, 0, Lm],
                         [Lm, 0, Lr, 0],
                         [0, Lm, 0, Lr]])

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
        Te = (3 * 4 / 4) * Lm * (iqs * idr - ids * iqr)

        return t_eval, Is_mag, rpm, Te
    except Exception as e:
        # en caso de fallo devolvemos arrays de referencia con el mismo tamaño
        n_points = n_points if isinstance(n_points, int) else 500
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        return t_eval, np.ones(n_points) * 1e6, np.zeros(n_points), np.zeros(n_points)

# -----------------------------------------------------------------------------
# Objective Function
# -----------------------------------------------------------------------------

class FullObjective:
    def __init__(self, target_current, target_rpm, target_torque):
        # target_* deben ser arrays con la misma longitud
        self.target_current = target_current
        self.target_rpm = target_rpm
        self.target_torque = target_torque
        self.eval_count = 0

    def __call__(self, params):
        self.eval_count += 1
        # penalizar parámetros no físicos
        if any(p <= 0 for p in params):
            return 1e10

        # simulate_motor devuelve t_eval y señales, ignoramos t_eval aquí
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

        print(f"     Iteración inicial - mejor costo: {gbest_cost:.6f}")

        vmax = (self.ub - self.lb) * 0.2

        for it in range(self.max_iter):
            # actualizar w
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

            if it % 10 == 0:
                print(f"     Iteración {it}/{self.max_iter} - mejor costo: {gbest_cost:.6f}")

        return gbest_cost, gbest


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
        alpha_pos = X[sorted_idx[0]].copy()
        beta_pos = X[sorted_idx[1]].copy()
        delta_pos = X[sorted_idx[2]].copy()

        print(f"     Iteración inicial - mejor costo: {fitness[sorted_idx[0]]:.6f}")

        for it in range(self.max_iter + 1):
            a = 2 - it * (2 / max(1, self.max_iter))
            for i in range(self.n_wolves):
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X[i])

                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - X[i])

                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X[i])

                X[i] = np.clip((X1 + X2 + X3) / 3, self.lb, self.ub)

            fitness = np.array([self.obj(x) for x in X])
            sorted_idx = np.argsort(fitness)
            alpha_pos, beta_pos, delta_pos = X[sorted_idx[0]], X[sorted_idx[1]], X[sorted_idx[2]]

            if it % 10 == 0:
                print(f"     Iteración {it}/{self.max_iter} - mejor costo: {fitness[sorted_idx[0]]:.6f}")

        return fitness[sorted_idx[0]], alpha_pos

# -----------------------------------------------------------------------------
# Guardar parámetros reales vs estimados
# -----------------------------------------------------------------------------

def guardar_parametros_csv(nombre_archivo, algoritmo, run_id, true_params, est_params, cost, error):
    data = {
        'algoritmo': algoritmo,
        'run': run_id,
        'costo': cost,
        'error_%': error
    }
    for i, name in enumerate(PARAM_NAMES):
        data[f'{name}_real'] = float(true_params[i])
        data[f'{name}_estimado'] = float(est_params[i])

    df = pd.DataFrame([data])
    if not os.path.exists(nombre_archivo):
        df.to_csv(nombre_archivo, index=False, mode='w')
    else:
        df.to_csv(nombre_archivo, index=False, mode='a', header=False)

# -----------------------------------------------------------------------------
# Guardar señales (tiempo + reales vs estimadas)
# -----------------------------------------------------------------------------

def guardar_senales_csv(nombre_archivo, algoritmo, run_id, t, señales_reales, señales_estimadas):
    """
    señales_reales / señales_estimadas: dict con keys 'corriente','rpm','torque' y arrays del mismo largo que t
    """
    df = pd.DataFrame({
        'algoritmo': [algoritmo] * len(t),
        'run': [run_id] * len(t),
        't': t,
        'corriente_real': señales_reales['corriente'],
        'rpm_real': señales_reales['rpm'],
        'torque_real': señales_reales['torque'],
        'corriente_estimado': señales_estimadas['corriente'],
        'rpm_estimado': señales_estimadas['rpm'],
        'torque_estimado': señales_estimadas['torque']
    })

    if not os.path.exists(nombre_archivo):
        df.to_csv(nombre_archivo, index=False, mode='w')
    else:
        df.to_csv(nombre_archivo, index=False, mode='a', header=False)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_best_result(best_params, algorithm_name, target_t, target_current, target_rpm, target_torque):
    # simular con best_params (obtiene t_sim que debería coincidir con target_t en discretización)
    t_sim, sim_current, sim_rpm, sim_torque = simulate_motor(best_params, t_span=[target_t[0], target_t[-1]], n_points=len(target_t))

    plt.figure(figsize=(21, 6))

    plt.subplot(1, 3, 1)
    plt.plot(target_t, target_current, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_current, '--', linewidth=2, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Corriente ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Corriente (A)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(target_t, target_rpm, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_rpm, '--', linewidth=2, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Velocidad ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (RPM)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(target_t, target_torque, '-', linewidth=2, label='Objetivo (Real)')
    plt.plot(t_sim, sim_torque, '--', linewidth=2, label=f'Estimado ({algorithm_name})')
    plt.title(f'Comparación de Torque ({algorithm_name})')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f'Mejor Resultado de Optimización con {algorithm_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f'results/best_result_{algorithm_name}.png'
    plt.savefig(plot_filename)
    print(f"   Gráfica del mejor resultado para {algorithm_name} guardada en: {plot_filename}")
    plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("="*60)
    print("COMPARACIÓN PSO vs GWO ")
    print("="*60)

    print("\n1. Generando datos de referencia...")
    # ahora simulate_motor devuelve t_eval también
    target_t, target_current, target_rpm, target_torque = simulate_motor(TRUE_PARAMS, t_span=[0, 2], n_points=500)

    noise_percentage = 0.02

    noise_c = np.random.normal(0, np.max(target_current) * noise_percentage, len(target_current))
    noise_r = np.random.normal(0, np.max(target_rpm) * noise_percentage, len(target_rpm))
    noise_t = np.random.normal(0, np.max(target_torque) * noise_percentage, len(target_torque))

    target_current_noisy = target_current + noise_c
    target_rpm_noisy = target_rpm + noise_r
    target_torque_noisy = target_torque + noise_t

    print(f"   Datos generados y ruido del {noise_percentage * 100}% aplicado.")

    results = []
    algorithms = {'PSO': SimplePSO, 'GWO': SimpleGWO}

    # Guardar la señal objetivo (solo una vez) para trazabilidad completa
    guardar_senales_csv(
        "results/target_signals.csv",
        algoritmo="target",
        run_id=0,
        t=target_t,
        señales_reales={'corriente': target_current, 'rpm': target_rpm, 'torque': target_torque},
        señales_estimadas={'corriente': target_current_noisy, 'rpm': target_rpm_noisy, 'torque': target_torque_noisy}
    )

    for alg_name, optimizer_class in algorithms.items():
        print(f"\n2. Ejecutando {alg_name}...")
        for run in range(N_RUNS):
            print(f"   Run {run+1}/{N_RUNS}:")
            start = time.time()
            obj = FullObjective(target_current_noisy, target_rpm_noisy, target_torque_noisy)
            optimizer = optimizer_class(obj, PARAM_BOUNDS)
            best_cost, best_params = optimizer.optimize()
            elapsed = time.time() - start

            param_error = np.mean(np.abs((best_params - TRUE_PARAMS) / TRUE_PARAMS)) * 100
            results.append({
                'algorithm': alg_name,
                'run': run + 1,
                'cost': float(best_cost),
                'param_error_%': float(param_error),
                'time_s': float(elapsed),
                'best_params': best_params
            })

            # Guardar en CSV parámetros reales vs estimados
            guardar_parametros_csv(
                "results/params_tracking.csv",
                alg_name,
                run + 1,
                TRUE_PARAMS,
                best_params,
                best_cost,
                param_error
            )

            # Simular con best_params para guardar las señales estimadas y compararlas
            t_sim, sim_current, sim_rpm, sim_torque = simulate_motor(best_params, t_span=[target_t[0], target_t[-1]], n_points=len(target_t))

            guardar_senales_csv(
                "results/signals_per_run.csv",
                algoritmo=alg_name,
                run_id=run + 1,
                t=t_sim,
                señales_reales={'corriente': target_current_noisy, 'rpm': target_rpm_noisy, 'torque': target_torque_noisy},
                señales_estimadas={'corriente': sim_current, 'rpm': sim_rpm, 'torque': sim_torque}
            )

            print(f"   Completado en {elapsed:.2f}s - Error parámetros: {param_error:.2f}%")

    df = pd.DataFrame(results)
    df.to_csv('results/comparison_improved.csv', index=False)

    print("\nRESULTADOS FINALES")
    print("="*60)
    summary = df.groupby('algorithm').agg({
        'param_error_%': ['mean', 'min'],
        'time_s': ['mean']
    }).reset_index()

    summary.columns = ['algorithm', 'error_promedio', 'mejor_error', 'tiempo_promedio']
    print(summary.to_string(index=False))

    # Graficar mejor resultado de cada algoritmo
    for alg_name in algorithms.keys():
        alg_data = df[df['algorithm'] == alg_name]
        best_run_idx = alg_data['param_error_%'].idxmin()
        best_run_params = alg_data.loc[best_run_idx, 'best_params']
        plot_best_result(best_run_params, alg_name, target_t, target_current_noisy, target_rpm_noisy, target_torque_noisy)

if __name__ == "__main__":
    main()
