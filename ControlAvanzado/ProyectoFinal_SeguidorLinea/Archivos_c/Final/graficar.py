#!/usr/bin/env python3
"""
Robot Log Visualizer - Graficador de datos del seguidor de línea
Procesa el log JSON y crea visualizaciones como la app
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class RobotLogVisualizer:
    def __init__(self, log_file_path):
        self.log_file = log_file_path
        self.data = []
        self.df = None
        
        # Configurar estilo de gráficas
        plt.style.use('dark_background')
        
    def load_data(self):
        """Cargar datos del archivo de log"""
        print("📊 Cargando datos del log...")
        
        try:
            with open(self.log_file, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:
                        try:
                            data_point = json.loads(line)
                            self.data.append(data_point)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Error en línea {line_num}: {e}")
                            continue
            
            # Convertir a DataFrame para fácil manejo
            self.df = pd.DataFrame(self.data)
            print(f"✅ Cargados {len(self.data)} puntos de datos")
            
        except FileNotFoundError:
            print(f"❌ No se encontró el archivo: {self.log_file}")
            return False
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
            
        return True
    
    def create_dashboard(self):
        """Crear dashboard completo con todas las gráficas"""
        if self.df is None or len(self.df) == 0:
            print("❌ No hay datos para graficar")
            return
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('🤖 Robot Seguidor de Línea - Dashboard de Telemetría', 
                     fontsize=16, fontweight='bold', color='white')
        
        # Layout de gráficas
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Posición de línea vs tiempo
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_line_position(ax1)
        
        # 2. Lazos de velocidad
        ax2 = fig.add_subplot(gs[1, 0])
        self.plot_velocity_loops(ax2)
        
        # 3. Errores de control
        ax3 = fig.add_subplot(gs[1, 1])
        self.plot_control_errors(ax3)
        
        # 4. Velocidad adaptativa
        ax4 = fig.add_subplot(gs[1, 2])
        self.plot_adaptive_speed(ax4)
        
        # 5. Trayectoria simulada
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_robot_trajectory(ax5)
        
        # 6. Estadísticas
        ax6 = fig.add_subplot(gs[2, 2])
        self.plot_statistics(ax6)
        
        plt.tight_layout()
        plt.show()
    
    def plot_line_position(self, ax):
        """Gráfica de posición de línea vs tiempo"""
        time = self.df['t']
        line_pos = self.df['line']
        
        # Colorear según intensidad del error
        colors = ['#10b981' if abs(x) <= 20 else '#f59e0b' if abs(x) <= 50 else '#ef4444' 
                 for x in line_pos]
        
        ax.scatter(time, line_pos, c=colors, s=2, alpha=0.7)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.axhline(y=50, color='#f59e0b', linestyle=':', alpha=0.5)
        ax.axhline(y=-50, color='#f59e0b', linestyle=':', alpha=0.5)
        
        ax.set_title('📍 Posición de Línea vs Tiempo', fontweight='bold')
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Posición Línea')
        ax.grid(True, alpha=0.3)
        
        # Leyenda de colores
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#10b981', markersize=8, label='Centrado (±20)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b', markersize=8, label='Desviado (±50)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444', markersize=8, label='Muy desviado')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    def plot_velocity_loops(self, ax):
        """Gráfica de lazos de velocidad"""
        time = self.df['t']
        
        ax.plot(time, self.df['vd_ref'], color='#ef4444', linewidth=1, label='VD Ref', alpha=0.8)
        ax.plot(time, self.df['vd_act'], color='#f59e0b', linewidth=1, label='VD Act', alpha=0.8)
        ax.plot(time, self.df['vi_ref'], color='#3b82f6', linewidth=1, label='VI Ref', alpha=0.8)
        ax.plot(time, self.df['vi_act'], color='#10b981', linewidth=1, label='VI Act', alpha=0.8)
        
        ax.set_title('⚡ Lazos de Velocidad', fontweight='bold')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Velocidad')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_control_errors(self, ax):
        """Gráfica de errores de control"""
        time = self.df['t']
        
        ax.plot(time, self.df['pos_err'], color='#ef4444', linewidth=1, label='Error Pos')
        ax.plot(time, self.df['vd_err'], color='#f59e0b', linewidth=1, label='Error VD', alpha=0.7)
        ax.plot(time, self.df['vi_err'], color='#10b981', linewidth=1, label='Error VI', alpha=0.7)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        ax.set_title('📊 Errores de Control', fontweight='bold')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_adaptive_speed(self, ax):
        """Gráfica de velocidad adaptativa"""
        time = self.df['t']
        va = self.df['va']
        
        ax.plot(time, va, color='#8b5cf6', linewidth=2, label='Velocidad Adaptativa')
        ax.fill_between(time, va, alpha=0.3, color='#8b5cf6')
        
        ax.set_title('🚀 Velocidad Adaptativa', fontweight='bold')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Velocidad (mm/s)')
        ax.grid(True, alpha=0.3)
        
        # Mostrar estadísticas
        ax.text(0.02, 0.98, f'Max: {va.max():.0f}\nMin: {va.min():.0f}\nPromedio: {va.mean():.0f}',
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                fontsize=8, color='white')
    
    def plot_robot_trajectory(self, ax):
        """Simular y graficar trayectoria del robot"""
        # Simulación simple de trayectoria basada en datos reales
        x, y = [0], [0]  # Posición inicial
        angle = 0        # Ángulo inicial
        
        for i in range(1, len(self.df)):
            dt = 0.048  # ~48ms entre muestras (basado en tu log)
            speed = self.df.iloc[i]['va'] * dt / 1000  # Convertir a metros
            line_error = self.df.iloc[i]['line']
            
            # Calcular cambio de ángulo basado en error de línea
            angle_change = -line_error * 0.001  # Factor de escala
            angle += angle_change
            
            # Calcular nueva posición
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        
        # Colorear trayectoria según error de línea
        colors = ['#10b981' if abs(err) <= 20 else '#f59e0b' if abs(err) <= 50 else '#ef4444' 
                 for err in self.df['line']]
        
        # Crear segmentos coloreados
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=2, alpha=0.8)
        
        # Marcar inicio y fin
        ax.plot(x[0], y[0], 'go', markersize=10, label='Inicio')
        ax.plot(x[-1], y[-1], 'ro', markersize=10, label='Fin')
        
        # Robot actual (última posición)
        robot = patches.Circle((x[-1], y[-1]), 0.05, color='#3b82f6', alpha=0.8)
        ax.add_patch(robot)
        
        ax.set_title('🗺️ Trayectoria Simulada del Robot', fontweight='bold')
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Información de la trayectoria
        total_distance = sum(np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2) for i in range(len(x)-1))
        ax.text(0.02, 0.98, f'Distancia total: {total_distance:.2f}m\nTiempo total: {len(self.df)*0.048:.1f}s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                fontsize=8, color='white')
    
    def plot_statistics(self, ax):
        """Mostrar estadísticas importantes"""
        # Calcular estadísticas
        stats = {
            'Tiempo total': f"{len(self.df) * 0.048:.1f}s",
            'Muestras': f"{len(self.df)}",
            'Error promedio': f"{abs(self.df['line']).mean():.1f}",
            'Error máximo': f"{abs(self.df['line']).max():.0f}",
            'Velocidad promedio': f"{self.df['va'].mean():.0f} mm/s",
            'Tiempo centrado': f"{(abs(self.df['line']) <= 20).sum() / len(self.df) * 100:.1f}%",
            'Tiempo desviado': f"{(abs(self.df['line']) > 50).sum() / len(self.df) * 100:.1f}%"
        }
        
        ax.axis('off')
        ax.set_title('📈 Estadísticas de Rendimiento', fontweight='bold', pad=20)
        
        y_pos = 0.9
        for key, value in stats.items():
            color = '#10b981' if 'centrado' in key or 'promedio' in key else '#f1f5f9'
            ax.text(0.1, y_pos, f"{key}:", fontweight='bold', color='#94a3b8', 
                   transform=ax.transAxes, fontsize=10)
            ax.text(0.6, y_pos, value, color=color, 
                   transform=ax.transAxes, fontsize=10, fontweight='bold')
            y_pos -= 0.12
    
    def create_analysis_report(self):
        """Crear reporte de análisis del comportamiento"""
        if self.df is None:
            return
        
        print("\n🤖 === REPORTE DE ANÁLISIS DEL ROBOT ===")
        print("=" * 50)
        
        # Análisis de seguimiento
        total_samples = len(self.df)
        centered_samples = (abs(self.df['line']) <= 20).sum()
        deviated_samples = (abs(self.df['line']) > 50).sum()
        
        print(f"📊 RENDIMIENTO DE SEGUIMIENTO:")
        print(f"   Tiempo total: {total_samples * 0.048:.1f} segundos")
        print(f"   Muestras totales: {total_samples}")
        print(f"   Tiempo centrado (±20): {centered_samples/total_samples*100:.1f}%")
        print(f"   Tiempo desviado (>50): {deviated_samples/total_samples*100:.1f}%")
        
        # Análisis de velocidad
        print(f"\n🚀 ANÁLISIS DE VELOCIDAD:")
        print(f"   Velocidad promedio: {self.df['va'].mean():.0f} mm/s")
        print(f"   Velocidad máxima: {self.df['va'].max():.0f} mm/s")
        print(f"   Velocidad mínima: {self.df['va'].min():.0f} mm/s")
        
        # Análisis de control
        print(f"\n🎯 ANÁLISIS DE CONTROL:")
        print(f"   Error posición promedio: {abs(self.df['pos_err']).mean():.1f}")
        print(f"   Error posición máximo: {abs(self.df['pos_err']).max():.0f}")
        print(f"   Error velocidad D promedio: {abs(self.df['vd_err']).mean():.1f}")
        print(f"   Error velocidad I promedio: {abs(self.df['vi_err']).mean():.1f}")
        
        # Detección de problemas
        print(f"\n⚠️  DETECCIÓN DE PROBLEMAS:")
        
        # Saturación de controladores
        vd_saturated = (abs(self.df['vd_out']) >= 2047).sum()
        vi_saturated = (abs(self.df['vi_out']) >= 2047).sum()
        
        if vd_saturated > total_samples * 0.1:
            print(f"   ⚠️  Motor derecho saturado {vd_saturated/total_samples*100:.1f}% del tiempo")
        if vi_saturated > total_samples * 0.1:
            print(f"   ⚠️  Motor izquierdo saturado {vi_saturated/total_samples*100:.1f}% del tiempo")
        
        # Pérdida de línea
        line_lost = (abs(self.df['line']) > 100).sum()
        if line_lost > 0:
            print(f"   ⚠️  Posible pérdida de línea en {line_lost} muestras")
        
        print("=" * 50)
    
    def run_analysis(self):
        """Ejecutar análisis completo"""
        if not self.load_data():
            return
        
        self.create_analysis_report()
        print("\n📊 Generando dashboard...")
        self.create_dashboard()

def main():
    """Función principal"""
    import sys
    
    # Verificar argumentos
    if len(sys.argv) != 2:
        print("Uso: python robot_visualizer.py <archivo_log.txt>")
        print("Ejemplo: python robot_visualizer.py serial_20250613_202615.txt")
        return
    
    log_file = sys.argv[1]
    
    # Crear visualizador y ejecutar análisis
    visualizer = RobotLogVisualizer(log_file)
    visualizer.run_analysis()

if __name__ == "__main__":
    main()