# maze_generator_32px_fix.py
# Genera laberintos 32x32 (exactamente 32x32 píxeles) siempre resolubles.
# Entrada = verde, Salida = rojo.

import numpy as np
import random
from typing import Tuple, List
from PIL import Image
import os
from collections import deque

def has_path(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Verifica si existe un camino entre start y end usando BFS"""
    n = maze.shape[0]
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        if (r, c) == end:
            return True
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and 
                maze[nr, nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    return False

def create_path(maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):
    """Crea un camino directo entre start y end si no existe"""
    sr, sc = start
    er, ec = end
    
    # Crear camino horizontal desde start hasta la columna de end
    current_c = sc
    while current_c != ec:
        if current_c < ec:
            current_c += 1
        else:
            current_c -= 1
        maze[sr, current_c] = 0
    
    # Crear camino vertical desde la posición actual hasta end
    current_r = sr
    while current_r != er:
        if current_r < er:
            current_r += 1
        else:
            current_r -= 1
        maze[current_r, ec] = 0

def generate_maze(n: int = 32) -> Tuple[np.ndarray, Tuple[int,int], Tuple[int,int]]:
    maze = np.ones((n, n), dtype=np.uint8)
    start = (1, 1)
    maze[start] = 0
    stack: List[Tuple[int, int]] = [start]
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def in_bounds(r: int, c: int) -> bool:
        return 1 <= r < n - 1 and 1 <= c < n - 1

    visited = set([start])

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and (nr, nc) not in visited:
                neighbors.append((nr, nc, dr, dc))

        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            wr, wc = r + dr // 2, c + dc // 2
            maze[wr, wc] = 0
            maze[nr, nc] = 0
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    # Entrada y salida
    entry = (1, 2)
    exit_ = (30, 30)

    # Asegurar conexión de entrada y salida
    maze[entry] = 0
    maze[exit_] = 0

    # Verificar si existe un camino, si no, crearlo
    if not has_path(maze, entry, exit_):
        create_path(maze, entry, exit_)

    return maze, entry, exit_


def save_maze_png(maze: np.ndarray, entry: Tuple[int, int], exit_: Tuple[int, int], filename: str):
    n = maze.shape[0]
    img = np.zeros((n, n, 3), dtype=np.uint8)

    img[maze == 1] = [0, 0, 0]       # pared = negro
    img[maze == 0] = [255, 255, 255] # camino = blanco
    img[entry] = [0, 255, 0]         # entrada = verde
    img[exit_] = [255, 0, 0]         # salida = rojo

    img = Image.fromarray(img, "RGB")
    img.save(filename)


if __name__ == "__main__":
    os.makedirs("mazes_32px", exist_ok=True)
    N = 32
    num_mazes = 10
    for i in range(num_mazes):
        maze, entry, exit_ = generate_maze(n=N)
        filename = f"mazes_32px/maze_{i+1:03d}.png"
        save_maze_png(maze, entry, exit_, filename)
    print(f"{num_mazes} laberintos de {N}x{N} píxeles guardados en 'mazes_32px/'")