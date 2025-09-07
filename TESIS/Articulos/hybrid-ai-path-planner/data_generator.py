"""
Generador de datos de laberintos usando DFS (Depth-First Search)
Para entrenar la DCGAN con laberintos válidos y solucionables
"""

import numpy as np
import random
from typing import Tuple, List
import os

class MazeDataGenerator:
    def __init__(self, size: int = 32):
        """
        Inicializa el generador de laberintos
        
        Args:
            size: Tamaño del laberinto (size x size)
        """
        self.size = size
        # El tamaño debe ser impar para el algoritmo DFS
        if size % 2 == 0:
            self.size = size - 1
    
    def generate_maze_dfs(self) -> np.ndarray:
        """
        Genera un laberinto usando algoritmo DFS
        
        Returns:
            maze: Array 2D donde 0=pared, 1=camino
        """
        # Inicializar con todas las paredes
        maze = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Stack para DFS y punto inicial
        stack = []
        start_x, start_y = 1, 1
        maze[start_y, start_x] = 1  # Marcar como camino
        stack.append((start_x, start_y))
        
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Movimientos de 2 en 2
        
        while stack:
            current_x, current_y = stack[-1]
            
            # Encontrar vecinos válidos
            neighbors = []
            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy
                
                # Verificar límites y que sea pared
                if (1 <= next_x < self.size - 1 and 
                    1 <= next_y < self.size - 1 and 
                    maze[next_y, next_x] == 0):
                    neighbors.append((next_x, next_y))
            
            if neighbors:
                # Elegir vecino aleatorio
                next_x, next_y = random.choice(neighbors)
                
                # Crear camino hacia el vecino
                wall_x = current_x + (next_x - current_x) // 2
                wall_y = current_y + (next_y - current_y) // 2
                
                maze[wall_y, wall_x] = 1
                maze[next_y, next_x] = 1
                
                stack.append((next_x, next_y))
            else:
                # Backtrack si no hay vecinos
                stack.pop()
        
        # Asegurar entrada y salida
        maze[1, 0] = 1  # Entrada
        maze[self.size-2, self.size-1] = 1  # Salida
        
        return maze
    
    def generate_dataset(self, num_mazes: int = 10000, 
                        save_path: str = None) -> np.ndarray:
        """
        Genera un dataset de laberintos
        
        Args:
            num_mazes: Número de laberintos a generar
            save_path: Ruta para guardar el dataset (opcional)
            
        Returns:
            dataset: Array con shape (num_mazes, size, size, 1)
        """
        print(f"Generando {num_mazes} laberintos de {self.size}x{self.size}...")
        
        dataset = np.zeros((num_mazes, self.size, self.size, 1), dtype=np.float32)
        
        for i in range(num_mazes):
            if i % 1000 == 0:
                print(f"Progreso: {i}/{num_mazes}")
            
            maze = self.generate_maze_dfs()
            dataset[i, :, :, 0] = maze.astype(np.float32)
        
        print("¡Generación completada!")
        
        if save_path:
            np.save(save_path, dataset)
            print(f"Dataset guardado en: {save_path}")
        
        return dataset
    
    def load_dataset(self, file_path: str) -> np.ndarray:
        """
        Carga un dataset previamente generado
        
        Args:
            file_path: Ruta del archivo .npy
            
        Returns:
            dataset: Array con los laberintos
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        dataset = np.load(file_path)
        print(f"Dataset cargado: {dataset.shape}")
        return dataset
    
    def preview_maze(self, maze: np.ndarray) -> None:
        """
        Imprime una vista previa del laberinto en consola
        
        Args:
            maze: Array 2D del laberinto
        """
        print("\nVista previa del laberinto:")
        print("█ = pared, · = camino")
        print("-" * (self.size + 2))
        
        for row in maze:
            line = "|"
            for cell in row:
                line += "·" if cell == 1 else "█"
            line += "|"
            print(line)
        
        print("-" * (self.size + 2))

# Ejemplo de uso
if __name__ == "__main__":
    # Crear generador
    generator = MazeDataGenerator(size=32)
    
    # Generar un laberinto de prueba
    test_maze = generator.generate_maze_dfs()
    generator.preview_maze(test_maze)
    
    # Generar dataset pequeño para pruebas
    small_dataset = generator.generate_dataset(num_mazes=100, 
                                              save_path="test_mazes.npy")
    
    print(f"Shape del dataset: {small_dataset.shape}")
    print(f"Tipo de datos: {small_dataset.dtype}")
    print(f"Rango de valores: [{small_dataset.min()}, {small_dataset.max()}]")
