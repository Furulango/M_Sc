"""
Visualizador para laberintos y métricas de entrenamiento
Monitoreo en tiempo real del progreso de la DCGAN
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional
import os
from datetime import datetime

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

class MazeVisualizer:
    def __init__(self, save_dir: str = "training_outputs"):
        """
        Inicializa el visualizador
        
        Args:
            save_dir: Directorio para guardar las visualizaciones
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Listas para almacenar métricas
        self.generator_losses = []
        self.discriminator_losses = []
        self.solvability_rates = []
        self.epochs = []
        
    def plot_maze(self, maze: np.ndarray, 
                  title: str = "Laberinto", 
                  path: Optional[List] = None,
                  save_path: Optional[str] = None) -> None:
        """
        Visualiza un solo laberinto
        
        Args:
            maze: Array 2D del laberinto
            title: Título de la figura
            path: Camino a mostrar (opcional)
            save_path: Ruta para guardar (opcional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Mostrar laberinto (0=negro/pared, 1=blanco/camino)
        ax.imshow(maze, cmap='gray_r', interpolation='nearest')
        
        # Mostrar camino si se proporciona
        if path:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, label='Camino')
            ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Inicio')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Fin')
            ax.legend()
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_maze_grid(self, mazes: np.ndarray, 
                      titles: Optional[List[str]] = None,
                      solvability: Optional[List[bool]] = None,
                      save_path: Optional[str] = None,
                      grid_size: tuple = (4, 4)) -> None:
        """
        Visualiza múltiples laberintos en una grilla
        
        Args:
            mazes: Array con shape (batch_size, height, width, channels)
            titles: Títulos para cada laberinto
            solvability: Lista indicando si cada laberinto es solucionable
            save_path: Ruta para guardar
            grid_size: Tamaño de la grilla (rows, cols)
        """
        rows, cols = grid_size
        num_mazes = min(len(mazes), rows * cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_mazes):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Convertir a 2D si es necesario
            maze_2d = mazes[i, :, :, 0] if len(mazes.shape) == 4 else mazes[i]
            
            # Mostrar laberinto
            ax.imshow(maze_2d, cmap='gray_r', interpolation='nearest')
            
            # Título con información de solvabilidad
            title = titles[i] if titles else f"Laberinto {i+1}"
            if solvability:
                status = "✓" if solvability[i] else "✗"
                color = "green" if solvability[i] else "red"
                title += f" {status}"
                ax.set_title(title, fontweight='bold', color=color)
            else:
                ax.set_title(title)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        
        # Ocultar axes vacíos
        for i in range(num_mazes, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_training_metrics(self, 
                            generator_losses: List[float],
                            discriminator_losses: List[float],
                            solvability_rates: List[float],
                            epochs: List[int],
                            save_path: Optional[str] = None) -> None:
        """
        Visualiza las métricas de entrenamiento
        
        Args:
            generator_losses: Pérdidas del generador
            discriminator_losses: Pérdidas del discriminador
            solvability_rates: Tasas de solvabilidad
            epochs: Épocas correspondientes
            save_path: Ruta para guardar
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pérdidas del generador y discriminador
        axes[0, 0].plot(epochs, generator_losses, 'b-', label='Generador', linewidth=2)
        axes[0, 0].plot(epochs, discriminator_losses, 'r-', label='Discriminador', linewidth=2)
        axes[0, 0].set_title('Pérdidas durante el Entrenamiento', fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Pérdida')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tasa de solvabilidad
        axes[0, 1].plot(epochs, solvability_rates, 'g-', linewidth=3)
        axes[0, 1].axhline(y=0.75, color='orange', linestyle='--', 
                          label='Objetivo (75%)', linewidth=2)
        axes[0, 1].set_title('Tasa de Solvabilidad', fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Tasa de Solvabilidad')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histograma de pérdidas recientes (últimas 10 épocas)
        recent_gen = generator_losses[-10:] if len(generator_losses) >= 10 else generator_losses
        recent_disc = discriminator_losses[-10:] if len(discriminator_losses) >= 10 else discriminator_losses
        
        axes[1, 0].hist(recent_gen, bins=20, alpha=0.7, label='Generador', color='blue')
        axes[1, 0].hist(recent_disc, bins=20, alpha=0.7, label='Discriminador', color='red')
        axes[1, 0].set_title('Distribución de Pérdidas (Últimas 10 Épocas)', fontweight='bold')
        axes[1, 0].set_xlabel('Pérdida')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Progreso hacia el objetivo
        if solvability_rates:
            current_rate = solvability_rates[-1]
            progress = min(current_rate / 0.75, 1.0) * 100  # Progreso hacia 75%
            
            axes[1, 1].pie([progress, 100-progress], 
                          labels=[f'Completado\n{current_rate:.1%}', f'Restante\n{1-current_rate:.1%}'],
                          colors=['lightgreen', 'lightcoral'],
                          autopct='%1.1f%%',
                          startangle=90)
            axes[1, 1].set_title('Progreso hacia Objetivo (75%)', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def update_metrics(self, 
                      epoch: int,
                      gen_loss: float,
                      disc_loss: float,
                      solvability_rate: float) -> None:
        """
        Actualiza las métricas almacenadas
        
        Args:
            epoch: Número de época
            gen_loss: Pérdida del generador
            disc_loss: Pérdida del discriminador
            solvability_rate: Tasa de solvabilidad
        """
        self.epochs.append(epoch)
        self.generator_losses.append(gen_loss)
        self.discriminator_losses.append(disc_loss)
        self.solvability_rates.append(solvability_rate)
    
    def save_training_summary(self, 
                            final_solvability: float,
                            total_epochs: int,
                            best_epoch: int,
                            training_time: float) -> None:
        """
        Crea y guarda un resumen final del entrenamiento
        
        Args:
            final_solvability: Tasa final de solvabilidad
            total_epochs: Número total de épocas
            best_epoch: Mejor época
            training_time: Tiempo total de entrenamiento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear resumen textual
        summary = f"""
=== RESUMEN DEL ENTRENAMIENTO ===
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RESULTADOS FINALES:
- Tasa de solvabilidad final: {final_solvability:.2%}
- Objetivo alcanzado (75%): {'✓ SÍ' if final_solvability >= 0.75 else '✗ NO'}
- Mejor época: {best_epoch}
- Épocas totales: {total_epochs}
- Tiempo de entrenamiento: {training_time:.2f} minutos

PÉRDIDAS FINALES:
- Generador: {self.generator_losses[-1]:.4f}
- Discriminador: {self.discriminator_losses[-1]:.4f}

PROGRESO:
- Solvabilidad inicial: {self.solvability_rates[0]:.2%}
- Solvabilidad final: {self.solvability_rates[-1]:.2%}
- Mejora total: {(self.solvability_rates[-1] - self.solvability_rates[0]):.2%}
"""
        
        # Guardar resumen
        summary_path = f"{self.save_dir}/training_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Crear y guardar gráfica final
        final_plot_path = f"{self.save_dir}/final_metrics_{timestamp}.png"
        self.plot_training_metrics(
            self.generator_losses,
            self.discriminator_losses,
            self.solvability_rates,
            self.epochs,
            save_path=final_plot_path
        )
        
        print(f"Resumen guardado en: {summary_path}")
        print(f"Gráficas finales guardadas en: {final_plot_path}")
    
    def plot_validation_stats(self, validation_results: Dict, 
                            save_path: Optional[str] = None) -> None:
        """
        Visualiza estadísticas de validación detalladas
        
        Args:
            validation_results: Resultados del validador
            save_path: Ruta para guardar
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribución de longitudes de camino
        if validation_results['path_lengths']:
            axes[0, 0].hist(validation_results['path_lengths'], bins=20, 
                           color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribución de Longitudes de Camino')
            axes[0, 0].set_xlabel('Longitud del Camino')
            axes[0, 0].set_ylabel('Frecuencia')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Distribución de ratios de paredes
        axes[0, 1].hist(validation_results['wall_ratios'], bins=20, 
                       color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribución de Ratios de Paredes')
        axes[0, 1].set_xlabel('Ratio de Paredes')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tasa de solvabilidad (gráfica de barras)
        solvable = validation_results['solvable_count']
        unsolvable = validation_results['total_mazes'] - solvable
        
        axes[1, 0].bar(['Solucionables', 'No Solucionables'], 
                      [solvable, unsolvable],
                      color=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title('Distribución de Solvabilidad')
        axes[1, 0].set_ylabel('Cantidad de Laberintos')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Estadísticas numéricas
        stats_text = f"""
        Estadísticas de Validación:
        
        Total de laberintos: {validation_results['total_mazes']}
        Solucionables: {validation_results['solvable_count']}
        Tasa de solvabilidad: {validation_results['solvability_rate']:.2%}
        
        Longitud promedio de camino: {validation_results['avg_path_length']:.1f}
        Ratio promedio de paredes: {validation_results['avg_wall_ratio']:.2%}
        
        Tiempo promedio de validación: {validation_results['avg_validation_time']:.4f}s
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear visualizador
    visualizer = MazeVisualizer()
    
    # Crear un laberinto de ejemplo
    test_maze = np.random.choice([0, 1], size=(32, 32), p=[0.6, 0.4])
    
    # Visualizar laberinto individual
    visualizer.plot_maze(test_maze, title="Laberinto de Prueba")
    
    # Crear múltiples laberintos para grilla
    test_mazes = np.random.choice([0, 1], size=(16, 32, 32, 1), p=[0.6, 0.4])
    solvability = np.random.choice([True, False], size=16, p=[0.7, 0.3])
    
    # Visualizar grilla
    visualizer.plot_maze_grid(test_mazes, 
                             solvability=solvability,
                             grid_size=(4, 4))
    
    print("Ejemplos de visualización creados!")
