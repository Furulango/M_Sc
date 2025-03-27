import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Cargar los datos desde el archivo .txt
def load_data(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    time = data[:, 0]
    input_signal = data[:, 1]
    output_signal = data[:, 2]
    return time, input_signal, output_signal

def calculate_parameter_entropy(population):
    parameter_entropies = []
    for param_column in population.T:
        hist, _ = np.histogram(param_column, bins=20, density=True)
        hist = hist + 1e-10
        param_entropy = entropy(hist)
        parameter_entropies.append(param_entropy)
    return {
        'wn_entropy': parameter_entropies[0],
        'z_entropy': parameter_entropies[1],
        'K_entropy': parameter_entropies[2],
        'total_entropy': np.mean(parameter_entropies)
    }

# Modelo correcto de la respuesta de un sistema de segundo orden
def transfer_function_response(t, wn, z, K):
    if z < 1.0:
        wd = wn * np.sqrt(1 - z**2)
        return K * (1 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z/np.sqrt(1-z**2)) * np.sin(wd * t)))
    elif z == 1.0:
        return K * (1 - (1 + wn * t) * np.exp(-wn * t))
    else:
        s1 = -wn * (z + np.sqrt(z**2 - 1))
        s2 = -wn * (z - np.sqrt(z**2 - 1))
        return K * (1 - (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s1 - s2))

# Función de error (MSE)
def fitness_function(params, time, output_signal):
    wn, z, K = params
    if wn <= 0 or z <= 0 or K <= 0:
        return np.inf
    model_response = transfer_function_response(time, wn, z, K)
    mse = np.mean((model_response - output_signal) ** 2)
    return mse

# Función de mutación mejorada
def mutate(individual, LowLim, UppLim, mutation_rate=0.2):
    mutated = individual.copy()
    for j in range(len(individual)):
        if np.random.rand() < mutation_rate:
            distance_to_bounds = min(mutated[j] - LowLim[j], UppLim[j] - mutated[j])
            max_mutation = min(0.2, distance_to_bounds)
            delta = np.random.uniform(-max_mutation, max_mutation) * mutated[j]
            mutated[j] += delta
            mutated[j] = np.clip(mutated[j], LowLim[j], UppLim[j])
    return mutated

# Cruce mejorado (SBX - Simulated Binary Crossover)
def sbx_crossover(parent1, parent2, eta=1):
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    for i in range(len(parent1)):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:
                if parent1[i] < parent2[i]:
                    y1, y2 = parent1[i], parent2[i]
                else:
                    y1, y2 = parent2[i], parent1[i]
                beta = 1.0 + (2.0 * (y1 - 0.0) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta + 1))
                rand = np.random.random()
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                offspring1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                offspring2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
        else:
            offspring1[i] = parent1[i]
            offspring2[i] = parent2[i]
    return offspring1, offspring2

# Algoritmo Genético mejorado
def improved_genetic_algorithm(time, output_signal, G=200, PopSize=80, Pm=0.3, TournSize=3, 
                               LowLim=[0.1, 0.01, 0.1], UppLim=[20, 2, 200], elite_size=2):
    population = np.zeros((PopSize, 3))
    for i in range(PopSize):
        for j in range(3):
            population[i, j] = LowLim[j] + np.random.random() * (UppLim[j] - LowLim[j])
    EntropiesHistory = {
        'wn_entropy': [],
        'z_entropy': [],
        'K_entropy': [],
        'total_entropy': []
    }
    BestFitnessHistory = np.zeros(G)
    AvgFitnessHistory = np.zeros(G)
    BestSolutionHistory = np.zeros((G, 3))
    stall_counter = 0
    best_fitness_ever = float('inf')
    for Iter in range(G):
        entropy_metrics = calculate_parameter_entropy(population)
        for key in EntropiesHistory:
            EntropiesHistory[key].append(entropy_metrics[key])
        Fitness = np.array([fitness_function(ind, time, output_signal) for ind in population])
        BestFitness = np.min(Fitness)
        AvgFitness = np.mean(Fitness)
        BestIdx = np.argmin(Fitness)
        BestInd = population[BestIdx].copy()
        BestFitnessHistory[Iter] = BestFitness
        AvgFitnessHistory[Iter] = AvgFitness
        BestSolutionHistory[Iter] = BestInd
        if abs(BestFitness - best_fitness_ever) < 1e-6:
            stall_counter += 1
        else:
            stall_counter = 0
            if BestFitness < best_fitness_ever:
                best_fitness_ever = BestFitness
        if stall_counter > 30:
            print(f"Convergencia alcanzada en la generación {Iter}")
            BestFitnessHistory = BestFitnessHistory[:Iter+1]
            AvgFitnessHistory = AvgFitnessHistory[:Iter+1]
            BestSolutionHistory = BestSolutionHistory[:Iter+1]
            break
        new_population = np.zeros((PopSize, 3))
        sorted_indices = np.argsort(Fitness)
        elite_indices = sorted_indices[:elite_size]
        for i in range(elite_size):
            new_population[i] = population[elite_indices[i]]
        for i in range(elite_size, PopSize, 2):
            candidates1 = np.random.choice(PopSize, TournSize, replace=False)
            parent1_idx = candidates1[np.argmin(Fitness[candidates1])]
            candidates2 = np.random.choice(PopSize, TournSize, replace=False)
            parent2_idx = candidates2[np.argmin(Fitness[candidates2])]
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            if i + 1 < PopSize:
                child1, child2 = sbx_crossover(parent1, parent2)
                new_population[i] = child1
                new_population[i+1] = child2
            else:
                new_population[i] = (parent1 + parent2) / 2
        for i in range(elite_size, PopSize):
            new_population[i] = mutate(new_population[i], LowLim, UppLim, Pm)
        population = new_population
        if Iter % 10 == 0:
            print(f"Generación {Iter}: Mejor MSE = {BestFitness}, Total Entropy = {entropy_metrics['total_entropy']:.4f}")
    return BestSolutionHistory, BestFitnessHistory, AvgFitnessHistory, EntropiesHistory

def plot_system_response(filename, wn, z, K, BestFitnessHistory):
    time, input_signal, output_signal = load_data(filename)
    output_max = np.max(output_signal)
    if output_max > 0:
        output_signal_norm = output_signal / output_max
        K_norm = K / output_max
    else:
        output_signal_norm = output_signal
        K_norm = K
    model_response = transfer_function_response(time, wn, z, K_norm)
    if output_max > 0:
        model_response = model_response * output_max
    residuals = output_signal - model_response
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    if z < 1.0:
        system_type = "Subamortiguado"
        wd = wn * np.sqrt(1 - z**2)
        tf_equation = f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*z*wn:.4f}s + {wn**2:.4f})"
        poles = f"Polos: {-z*wn:.4f} ± j{wd:.4f}"
    elif z == 1.0:
        system_type = "Críticamente amortiguado"
        tf_equation = f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*wn:.4f}s + {wn**2:.4f})"
        poles = f"Polos: s = {-wn:.4f} (doble)"
    else:
        system_type = "Sobreamortiguado"
        p1 = -wn * (z + np.sqrt(z**2 - 1))
        p2 = -wn * (z - np.sqrt(z**2 - 1))
        tf_equation = f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*z*wn:.4f}s + {wn**2:.4f})"
        poles = f"Polos: s = {p1:.4f}, s = {p2:.4f}"
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 3, 1)
    plt.plot(time, output_signal, 'b-', linewidth=2, label='Datos reales')
    plt.plot(time, model_response, 'r--', linewidth=2, label='Modelo ajustado')
    plt.title(f'Respuesta del Sistema\n(wn={wn:.4f}, z={z:.4f}, K={K:.4f})')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 3, 2)
    plt.plot(time, input_signal, 'g-', linewidth=2)
    plt.title('Señal de Entrada')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.subplot(2, 3, 3)
    plt.plot(time, residuals, 'r-', linewidth=1)
    plt.title('Residuos (Error)')
    plt.xlabel('Tiempo')
    plt.ylabel('Error')
    plt.grid(True)
    plt.subplot(2, 3, 4)
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Distribución de Residuos')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.subplot(2, 3, 5)
    plt.plot(BestFitnessHistory, 'b-', label='Mejor MSE')
    plt.title('Evolución del Fitness')
    plt.xlabel('Generación')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    results_text = (
        f"Resultados del Sistema:\n\n"
        f"Parámetros:\n"
        f"Wn = {wn:.4f}\n"
        f"Z = {z:.4f}\n"
        f"K = {K:.4f}\n\n"
        f"Métricas de Rendimiento:\n"
        f"MSE: {mse:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n\n"
        f"Tipo de Sistema: {system_type}\n\n"
        f"Función de Transferencia:\n"
        f"{tf_equation}\n\n"
        f"{poles}"
    )
    plt.text(0.05, 0.95, results_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()

def plot_entropy_evolution(EntropiesHistory):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(EntropiesHistory['wn_entropy'], label='Entropía Frecuencia Natural (wn)')
    plt.title('Entropía de Frecuencia Natural')
    plt.xlabel('Generación')
    plt.ylabel('Entropía')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(EntropiesHistory['z_entropy'], label='Entropía Ratio Amortiguamiento (z)', color='green')
    plt.title('Entropía de Ratio de Amortiguamiento')
    plt.xlabel('Generación')
    plt.ylabel('Entropía')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(EntropiesHistory['K_entropy'], label='Entropía Ganancia (K)', color='red')
    plt.title('Entropía de Ganancia')
    plt.xlabel('Generación')
    plt.ylabel('Entropía')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(EntropiesHistory['total_entropy'], label='Entropía Total', color='purple')
    plt.title('Entropía Total de la Población')
    plt.xlabel('Generación')
    plt.ylabel('Entropía')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Función principal para cargar los datos, ejecutar el AG y mostrar resultados
def main(filename):
    try:
        time, _, output_signal = load_data(filename)
        print(f"Datos cargados exitosamente: {len(time)} puntos")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return
    output_max = np.max(output_signal)
    if output_max > 0:
        output_signal_norm = output_signal / output_max
    else:
        output_signal_norm = output_signal
    LowLim = [0.1, 0.01, 0.1]
    UppLim = [20, 2, 2]
    print("Algoritmo genético...")
    BestSolutionHistory, BestFitnessHistory, AvgFitnessHistory, EntropiesHistory = improved_genetic_algorithm(
        time, output_signal_norm, G=200, PopSize=80, LowLim=LowLim, UppLim=UppLim)
    BestParams = BestSolutionHistory[-1]
    wn, z, K = BestParams
    plot_entropy_evolution(EntropiesHistory)
    if output_max > 0:
        K = K * output_max
        BestParams[2] = K
    plot_system_response(filename, wn, z, K, BestFitnessHistory)
    if z < 1.0:
        system_type = "Subamortiguado"
    elif z == 1.0:
        system_type = "Críticamente amortiguado"
    else:
        system_type = "Sobreamortiguado"
    print(f"Tipo de sistema: {system_type}")
    if z < 1.0:
        wd = wn * np.sqrt(1 - z**2)
        print(f"\nFunción de transferencia aproximada:")
        print(f"G(s) = {K:.4f} * ω²/(s² + 2ζωₙs + ω²)")
        print(f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*z*wn:.4f}s + {wn**2:.4f})")
        print(f"\nPolos: {-z*wn:.4f} ± j{wd:.4f}")
    elif z == 1.0:
        print(f"\nFunción de transferencia aproximada:")
        print(f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*wn:.4f}s + {wn**2:.4f})")
        print(f"\nPolos: s = {-wn:.4f} (doble)")
    else:
        p1 = -wn * (z + np.sqrt(z**2 - 1))
        p2 = -wn * (z - np.sqrt(z**2 - 1))
        print(f"\nFunción de transferencia aproximada:")
        print(f"G(s) = {K:.4f} * {wn**2:.4f}/(s² + {2*z*wn:.4f}s + {wn**2:.4f})")
        print(f"\nPolos: s = {p1:.4f}, s = {p2:.4f}")
    model_response = transfer_function_response(time, wn, z, K if output_max <= 0 else K/output_max)
    if output_max > 0:
        model_response = model_response * output_max
