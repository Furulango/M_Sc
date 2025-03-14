import numpy as np
import matplotlib.pyplot as plt

def enhanced_genetic_algorithm(G=100, PopSize=50, Dv=1, n=14, Pm=0.2, TournSize=2, OffsProb=0.5, LowLim=-100, UppLim=100):
    # Objective Function
    def objective_function(x):
        return -(0.001 * x**2) + 3.7 + np.cos(x / 4 + np.pi / 6)

    # Generate initial population (corrected range)
    IniPop = np.random.randint(0, 2**n, (PopSize, Dv))

    # Initialize variables
    BestFitnessHistory = np.zeros(G)
    BestSolutionHistory = np.zeros(G)

    for Iter in range(G):
        # Decode and evaluate fitness
        IndDecoded = LowLim + (UppLim - LowLim) * (IniPop / (2**n - 1))
        Fitness = objective_function(IndDecoded)
        
        # Track best individual
        BestFitness = np.max(Fitness)
        BestIdx = np.argmax(Fitness)
        BestInd = IniPop[BestIdx]
        BestFitnessHistory[Iter] = BestFitness
        BestSolutionHistory[Iter] = IndDecoded[BestIdx]
        
        # Tournament selection and crossover
        Offspring = np.zeros(PopSize, dtype=int)
        for i in range(PopSize):
            # Select parents using tournament selection
            candidates = np.random.choice(PopSize, TournSize, replace=False)
            parent1 = IniPop[candidates[np.argmax(Fitness[candidates])]]
            
            candidates = np.random.choice(PopSize, TournSize, replace=False)
            parent2 = IniPop[candidates[np.argmax(Fitness[candidates])]]
            
            # Single-point crossover
            Cp = np.random.randint(1, n)
            maskT = (2**Cp) - 1
            maskH = (2**n - 1) - maskT
            
            head1 = parent1 & maskH
            tail1 = parent1 & maskT
            head2 = parent2 & maskH
            tail2 = parent2 & maskT
            
            if np.random.rand() < OffsProb:
                Offspring[i] = head1 + tail2
            else:
                Offspring[i] = head2 + tail1
        
        # Mutation (bit-wise)
        for j in range(PopSize):
            if np.random.rand() < Pm:
                bitPos = np.random.randint(0, n)
                Offspring[j] = np.bitwise_xor(Offspring[j], 2**bitPos)
        
        # Elitism: preserve best individual
        worstIdx = np.argmin(Fitness)
        Offspring[worstIdx] = BestInd
        
        # Update population for next generation
        IniPop = Offspring

    # Best solution
    BestSolution = BestSolutionHistory[-1]
    BestFitnessValue = BestFitnessHistory[-1]

    # Display results
    print(f"Best Solution: {BestSolution}")
    print(f"Best Fitness: {BestFitnessValue}")

    # Plot convergence
    plt.figure()
    plt.plot(BestFitnessHistory, linewidth=2)
    plt.title('Best Fitness per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)

    plt.figure()
    plt.plot(BestSolutionHistory, linewidth=2)
    plt.title('Best Solution per Generation')
    plt.xlabel('Generation')
    plt.ylabel('x value')
    plt.grid(True)

    # Plot final solution on objective function
    x = np.linspace(LowLim, UppLim, 1000)
    y = objective_function(x)

    plt.figure()
    plt.plot(x, y, 'k')
    plt.plot(BestSolution, BestFitnessValue, 'ro', markersize=10, linewidth=2)
    plt.title('Objective Function with Optimal Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Function', 'Optimal Solution'])
    plt.show()

    return BestSolution, BestFitnessValue


# Run the enhanced genetic algorithm
enhanced_genetic_algorithm()
