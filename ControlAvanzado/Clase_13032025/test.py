import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo .txt
def load_data(filename):
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    time = data[:, 0]
    input_signal = data[:, 1]
    output_signal = data[:, 2]
    return time, input_signal, output_signal

# Modelo correcto de la respuesta de un sistema de segundo orden
def transfer_function_response(t, wn, z, K):
    if z < 1.0:  # Subamortiguado
        wd = wn * np.sqrt(1 - z**2)
        return K * (1 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z/np.sqrt(1-z**2)) * np.sin(wd * t)))
    elif z == 1.0:  # Críticamente amortiguado
        return K * (1 - (1 + wn * t) * np.exp(-wn * t))
    else:  # Sobreamortiguado
        s1 = -wn * (z + np.sqrt(z**2 - 1))
        s2 = -wn * (z - np.sqrt(z**2 - 1))
        return K * (1 - (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s1 - s2))

# Función de error (MSE)
def fitness_function(params, time, output_signal):
    wn, z, K = params
    # Para evitar problemas con valores inválidos
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
            # Mutación adaptativa - más pequeña cerca de los límites
            distance_to_bounds = min(mutated[j] - LowLim[j], UppLim[j] - mutated[j])
            max_mutation = min(0.2, distance_to_bounds) # Limita la mutación a 20% o menos
            delta = np.random.uniform(-max_mutation, max_mutation) * mutated[j]
            mutated[j] += delta
            
            # Asegurarse de que el parámetro esté dentro de los límites
            mutated[j] = np.clip(mutated[j], LowLim[j], UppLim[j])
    return mutated

# Cruce mejorado (SBX - Simulated Binary Crossover)
def sbx_crossover(parent1, parent2, eta=1):
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:  # Evitar división por cero
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
    # Inicializar la población directamente con valores reales
    population = np.zeros((PopSize, 3))
    for i in range(PopSize):
        for j in range(3):
            population[i, j] = LowLim[j] + np.random.random() * (UppLim[j] - LowLim[j])
    
    BestFitnessHistory = np.zeros(G)
    AvgFitnessHistory = np.zeros(G)
    BestSolutionHistory = np.zeros((G, 3))
    
    # Para convergencia temprana
    stall_counter = 0
    best_fitness_ever = float('inf')
    
    for Iter in range(G):
        # Evaluar la aptitud (función de error)
        Fitness = np.array([fitness_function(ind, time, output_signal) for ind in population])
        
        # Rastrear el mejor individuo y el promedio
        BestFitness = np.min(Fitness)
        AvgFitness = np.mean(Fitness)
        BestIdx = np.argmin(Fitness)
        BestInd = population[BestIdx].copy()
        
        BestFitnessHistory[Iter] = BestFitness
        AvgFitnessHistory[Iter] = AvgFitness
        BestSolutionHistory[Iter] = BestInd
        
        # Verificar convergencia
        if abs(BestFitness - best_fitness_ever) < 1e-6:
            stall_counter += 1
        else:
            stall_counter = 0
            if BestFitness < best_fitness_ever:
                best_fitness_ever = BestFitness
        
        # Si no hay mejora durante 30 generaciones, terminar
        if stall_counter > 30:
            print(f"Convergencia alcanzada en la generación {Iter}")
            # Truncar las historias
            BestFitnessHistory = BestFitnessHistory[:Iter+1]
            AvgFitnessHistory = AvgFitnessHistory[:Iter+1]
            BestSolutionHistory = BestSolutionHistory[:Iter+1]
            break
        
        # Crear nueva población con elitismo
        new_population = np.zeros((PopSize, 3))
        
        # Preservar los mejores individuos (elitismo)
        sorted_indices = np.argsort(Fitness)
        elite_indices = sorted_indices[:elite_size]
        for i in range(elite_size):
            new_population[i] = population[elite_indices[i]]
        
        # Selección y reproducción para el resto de la población
        for i in range(elite_size, PopSize, 2):
            # Selección por torneo
            candidates1 = np.random.choice(PopSize, TournSize, replace=False)
            parent1_idx = candidates1[np.argmin(Fitness[candidates1])]
            candidates2 = np.random.choice(PopSize, TournSize, replace=False)
            parent2_idx = candidates2[np.argmin(Fitness[candidates2])]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Cruce
            if i + 1 < PopSize:  # Asegurarse de que hay espacio para dos descendientes
                child1, child2 = sbx_crossover(parent1, parent2)
                new_population[i] = child1
                new_population[i+1] = child2
            else:
                # En caso de población impar
                new_population[i] = (parent1 + parent2) / 2
        
        # Mutación (excluyendo a la élite)
        for i in range(elite_size, PopSize):
            new_population[i] = mutate(new_population[i], LowLim, UppLim, Pm)
        
        # Actualizar la población
        population = new_population
        
        # Imprimir progreso cada 10 generaciones
        if Iter % 10 == 0:
            print(f"Generación {Iter}: Mejor MSE = {BestFitness}, wn = {BestInd[0]:.4f}, z = {BestInd[1]:.4f}, K = {BestInd[2]:.4f}")
    
    return BestSolutionHistory, BestFitnessHistory, AvgFitnessHistory

# Función principal para cargar los datos, ejecutar el AG y mostrar resultados
def main(filename):
    # Cargar los datos
    try:
        time, _, output_signal = load_data(filename)
        print(f"Datos cargados exitosamente: {len(time)} puntos")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return
    
    # Normalizar la salida si es necesario
    output_max = np.max(output_signal)
    if output_max > 0:
        output_signal_norm = output_signal / output_max
    else:
        output_signal_norm = output_signal
        
    # Parámetros iniciales para el AG
    LowLim = [0.1, 0.01, 0.1]   # Límites inferiores para [wn, z, K]
    UppLim = [20, 2, 2]         # Límites superiores para [wn, z, K]
    
    print("Ejecutando algoritmo genético...")
    BestSolutionHistory, BestFitnessHistory, AvgFitnessHistory = improved_genetic_algorithm(
        time, output_signal_norm, G=200, PopSize=80, LowLim=LowLim, UppLim=UppLim)
    
    # Mejores parámetros encontrados
    BestParams = BestSolutionHistory[-1]
    wn, z, K = BestParams
    
    # Si se normalizó la salida, ajustar K
    if output_max > 0:
        K = K * output_max
        BestParams[2] = K
    
    print("\nResultados finales:")
    print(f"Wn = {wn:.4f}")
    print(f"Z = {z:.4f}")
    print(f"K = {K:.4f}")
    print(f"MSE = {BestFitnessHistory[-1]:.6f}")
    
    # Tipo de respuesta del sistema
    if z < 1.0:
        system_type = "Subamortiguado"
    elif z == 1.0:
        system_type = "Críticamente amortiguado"
    else:
        system_type = "Sobreamortiguado"
    print(f"Tipo de sistema: {system_type}")
    
    # Modelo en función de transferencia
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
    
    # Calcular la respuesta del modelo con los mejores parámetros
    model_response = transfer_function_response(time, wn, z, K if output_max <= 0 else K/output_max)
    if output_max > 0:
        model_response = model_response * output_max
    
    # Calcular métricas de rendimiento
    mse = np.mean((model_response - output_signal) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(model_response - output_signal))
    print(f"\nMétricas de rendimiento:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Graficar la evolución de la aptitud (MSE) y la mejor solución por generación
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(BestFitnessHistory) + 1), BestFitnessHistory, 'b-', linewidth=2, label='Mejor aptitud')
    plt.plot(np.arange(1, len(AvgFitnessHistory) + 1), AvgFitnessHistory, 'r--', linewidth=1, label='Aptitud promedio')
    plt.title('Evolución de la aptitud (MSE)')
    plt.xlabel('Generación')
    plt.ylabel('Error cuadrático medio (MSE)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala logarítmica para mejor visualización
    
    if 'BestSolutionHistory' in locals():
        plt.subplot(1, 2, 2)
        generations = np.arange(1, len(BestSolutionHistory) + 1)
        plt.plot(generations, BestSolutionHistory[:, 0], 'r-', label='wn')
        plt.plot(generations, BestSolutionHistory[:, 1], 'g-', label='z')
        plt.plot(generations, BestSolutionHistory[:, 2], 'b-', label='K')
        plt.title('Evolución de los parámetros')
        plt.xlabel('Generación')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
    else:
        plt.figure(figsize=(10, 6))
        if 'time' in locals() and 'output_signal' in locals() and 'model_response' in locals():
            plt.plot(time, output_signal, 'b-', linewidth=2, label='Datos reales')
        plt.plot(time, model_response, 'r--', linewidth=2, label='Modelo ajustado')
        if 'wn' in locals() and 'z' in locals() and 'K' in locals():
            plt.title(f'Respuesta del Sistema: Real vs Modelo (wn={wn:.4f}, z={z:.4f}, K={K:.4f})')
        else:
            plt.title('Respuesta del Sistema: Real vs Modelo')
        plt.xlabel('Tiempo')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.grid(True)
    print("Error: Variables 'time', 'output_signal', or 'model_response' are not defined.")

    if 'output_signal' in locals() and 'model_response' in locals() and 'time' in locals():
        residuals = output_signal - model_response
        plt.figure(figsize=(10, 6))
        plt.plot(time, residuals, 'g-', linewidth=1)
        plt.title('Residuos (Error)')
        plt.xlabel('Tiempo')
        plt.ylabel('Error')
        plt.grid(True)
    else:
        print("Error: Variables 'output_signal', 'model_response', or 'time' are not defined.")
    plt.xlabel('Tiempo')
    plt.ylabel('Error')
    plt.grid(True)

    plt.show()

# Ejecutar el código con el archivo de datos
if __name__ == "__main__":
    import os
    file_path = input("Ingrese la ruta del archivo de datos: ")
    if os.path.exists(file_path):
        main(file_path)
    else:
        print(f"El archivo {file_path} no existe.")