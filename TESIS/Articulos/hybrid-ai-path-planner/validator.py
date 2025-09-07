"""
Validador de laberintos usando BFS (Breadth-First Search)
Verifica que los laberintos generados sean solucionables
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional
import time

class MazeValidator:
    def __init__(self):
        """
        Inicializa el validador de laberintos
        """
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # derecha, abajo, izquierda, arriba
    
    def find_entrance_exit(self, maze: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Encuentra los puntos de entrada y salida del laberinto
        
        Args:
            maze: Array 2D del laberinto (0=pared, 1=camino)
            
        Returns:
            entrance, exit: Coordenadas (y, x) de entrada y salida
        """
        size = maze.shape[0]
        
        # Buscar entrada en el borde izquierdo
        entrance = None
        for y in range(size):
            if maze[y, 0] == 1:  # Es camino
                entrance = (y, 0)
                break
        
        # Si no hay entrada en borde izquierdo, usar (1,1) por defecto
        if entrance is None:
            entrance = (1, 1) if maze[1, 1] == 1 else (0, 0)
        
        # Buscar salida en el borde derecho
        exit_point = None
        for y in range(size):
            if maze[y, size-1] == 1:  # Es camino
                exit_point = (y, size-1)
                break
        
        # Si no hay salida en borde derecho, usar esquina opuesta
        if exit_point is None:
            exit_point = (size-2, size-2) if maze[size-2, size-2] == 1 else (size-1, size-1)
        
        return entrance, exit_point
    
    def bfs_path_exists(self, maze: np.ndarray, 
                       start: Tuple[int, int], 
                       end: Tuple[int, int]) -> bool:
        """
        Verifica si existe un camino entre start y end usando BFS
        
        Args:
            maze: Array 2D del laberinto
            start: Punto de inicio (y, x)
            end: Punto final (y, x)
            
        Returns:
            bool: True si existe camino, False si no
        """
        if maze[start[0], start[1]] == 0 or maze[end[0], end[1]] == 0:
            return False  # Start o end son paredes
        
        rows, cols = maze.shape
        visited = np.zeros_like(maze, dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True
        
        while queue:
            y, x = queue.popleft()
            
            # Si llegamos al destino
            if (y, x) == end:
                return True
            
            # Explorar vecinos
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                
                # Verificar límites
                if (0 <= ny < rows and 0 <= nx < cols and 
                    not visited[ny, nx] and maze[ny, nx] == 1):
                    
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        
        return False
    
    def bfs_find_path(self, maze: np.ndarray, 
                     start: Tuple[int, int], 
                     end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Encuentra el camino más corto usando BFS
        
        Args:
            maze: Array 2D del laberinto
            start: Punto de inicio (y, x)
            end: Punto final (y, x)
            
        Returns:
            path: Lista de coordenadas del camino, None si no existe
        """
        if maze[start[0], start[1]] == 0 or maze[end[0], end[1]] == 0:
            return None
        
        rows, cols = maze.shape
        visited = np.zeros_like(maze, dtype=bool)
        parent = {}
        queue = deque([start])
        visited[start[0], start[1]] = True
        parent[start] = None
        
        while queue:
            y, x = queue.popleft()
            
            # Si llegamos al destino, reconstruir camino
            if (y, x) == end:
                path = []
                current = end
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]  # Invertir para tener start->end
            
            # Explorar vecinos
            for dy, dx in self.directions:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < rows and 0 <= nx < cols and 
                    not visited[ny, nx] and maze[ny, nx] == 1):
                    
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (y, x)
                    queue.append((ny, nx))
        
        return None
    
    def validate_maze(self, maze: np.ndarray) -> dict:
        """
        Valida completamente un laberinto
        
        Args:
            maze: Array 2D del laberinto
            
        Returns:
            dict: Resultados de la validación
        """
        start_time = time.time()
        
        # Encontrar entrada y salida
        entrance, exit_point = self.find_entrance_exit(maze)
        
        # Verificar si existe camino
        has_solution = self.bfs_path_exists(maze, entrance, exit_point)
        
        # Si tiene solución, encontrar el camino
        path = None
        path_length = 0
        if has_solution:
            path = self.bfs_find_path(maze, entrance, exit_point)
            path_length = len(path) if path else 0
        
        # Calcular estadísticas
        total_cells = maze.size
        wall_cells = np.sum(maze == 0)
        path_cells = np.sum(maze == 1)
        wall_ratio = wall_cells / total_cells
        
        validation_time = time.time() - start_time
        
        return {
            'is_solvable': has_solution,
            'entrance': entrance,
            'exit': exit_point,
            'path_length': path_length,
            'path': path,
            'wall_ratio': wall_ratio,
            'total_cells': total_cells,
            'wall_cells': wall_cells,
            'path_cells': path_cells,
            'validation_time': validation_time
        }
    
    def validate_batch(self, mazes: np.ndarray, verbose: bool = True) -> dict:
        """
        Valida un batch de laberintos
        
        Args:
            mazes: Array con shape (batch_size, height, width, channels)
            verbose: Mostrar progreso
            
        Returns:
            dict: Estadísticas del batch
        """
        batch_size = mazes.shape[0]
        solvable_count = 0
        total_validation_time = 0
        path_lengths = []
        wall_ratios = []
        
        if verbose:
            print(f"Validando {batch_size} laberintos...")
        
        for i in range(batch_size):
            if verbose and i % 100 == 0:
                print(f"Progreso: {i}/{batch_size}")
            
            # Convertir a 2D si es necesario
            maze_2d = mazes[i, :, :, 0] if len(mazes.shape) == 4 else mazes[i]
            
            # Validar
            result = self.validate_maze(maze_2d)
            
            if result['is_solvable']:
                solvable_count += 1
                path_lengths.append(result['path_length'])
            
            wall_ratios.append(result['wall_ratio'])
            total_validation_time += result['validation_time']
        
        # Calcular estadísticas
        solvability_rate = solvable_count / batch_size
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        avg_wall_ratio = np.mean(wall_ratios)
        avg_validation_time = total_validation_time / batch_size
        
        if verbose:
            print(f"Validación completada!")
            print(f"Tasa de solvabilidad: {solvability_rate:.2%}")
        
        return {
            'solvability_rate': solvability_rate,
            'solvable_count': solvable_count,
            'total_mazes': batch_size,
            'avg_path_length': avg_path_length,
            'avg_wall_ratio': avg_wall_ratio,
            'avg_validation_time': avg_validation_time,
            'path_lengths': path_lengths,
            'wall_ratios': wall_ratios
        }
    
    def repair_maze(self, maze: np.ndarray) -> np.ndarray:
        """
        Intenta reparar un laberinto no solucionable
        Conecta entrada y salida con un camino directo
        
        Args:
            maze: Array 2D del laberinto
            
        Returns:
            repaired_maze: Laberinto reparado
        """
        repaired = maze.copy()
        entrance, exit_point = self.find_entrance_exit(maze)
        
        # Si ya es solucionable, no hacer nada
        if self.bfs_path_exists(maze, entrance, exit_point):
            return repaired
        
        # Crear camino directo (línea recta horizontal o vertical)
        y1, x1 = entrance
        y2, x2 = exit_point
        
        # Camino horizontal primero, luego vertical
        for x in range(min(x1, x2), max(x1, x2) + 1):
            repaired[y1, x] = 1
        
        for y in range(min(y1, y2), max(y1, y2) + 1):
            repaired[y, x2] = 1
        
        return repaired

# Ejemplo de uso
if __name__ == "__main__":
    # Crear validador
    validator = MazeValidator()
    
    # Crear un laberinto de prueba simple
    test_maze = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    
    print("Laberinto de prueba:")
    print(test_maze)
    
    # Validar
    result = validator.validate_maze(test_maze)
    print(f"\nResultados de validación:")
    print(f"¿Es solucionable?: {result['is_solvable']}")
    print(f"Entrada: {result['entrance']}")
    print(f"Salida: {result['exit']}")
    print(f"Longitud del camino: {result['path_length']}")
    print(f"Ratio de paredes: {result['wall_ratio']:.2%}")
    
    if result['path']:
        print(f"Camino: {result['path']}")
