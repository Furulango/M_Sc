#!/usr/bin/env python3
"""
Robot Log Visualizer - Graficador de datos del seguidor de línea
Procesa el log JSON y crea visualizaciones con análisis RMS de lazos de control
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
        self.rms_metrics = {}
        
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
            
            # Calcular métricas RMS
            self.calculate_rms_metrics()
            
        except FileNotFoundError:
            print(f"❌ No se encontró el archivo: {self.log_file}")
            return False
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
            
        return True
    
    def calculate_rms_metrics(self):
        """Calcular métricas RMS para todos los lazos de control"""
        print("🔢 Calculando métricas RMS...")
        
        # RMS de errores de velocidad
        vd_error = self.df['vd_ref'] - self.df['vd_act']
        vi_error = self.df['vi_ref'] - self.df['vi_act']
        
        # RMS del error de posición (ya viene calculado)
        pos_error = self.df['pos_err']
        
        self.rms_metrics = {
            # Errores RMS
            'vd_error_rms': np.sqrt(np.mean(vd_error**2)),
            'vi_error_rms': np.sqrt(np.mean(vi_error**2)),
            'pos_error_rms': np.sqrt(np.mean(pos_error**2)),
            
            # RMS de referencias (para normalización)
            'vd_ref_rms': np.sqrt(np.mean(self.df['vd_ref']**2)),
            'vi_ref_rms': np.sqrt(np.mean(self.df['vi_ref']**2)),
            
            # RMS de retroalimentación
            'vd_act_rms': np.sqrt(np.mean(self.df['vd_act']**2)),
            'vi_act_rms': np.sqrt(np.mean(self.df['vi_act']**2)),
            
            # Porcentaje de error RMS respecto a la referencia
            'vd_error_percent': (np.sqrt(np.mean(vd_error**2)) / np.sqrt(np.mean(self.df['vd_ref']**2))) * 100 if np.sqrt(np.mean(self.df['vd_ref']**2)) > 0 else 0,
            'vi_error_percent': (np.sqrt(np.mean(vi_error**2)) / np.sqrt(np.mean(self.df['vi_ref']**2))) * 100 if np.sqrt(np.mean(self.df['vi_ref']**2)) > 0 else 0,
            
            # Correlación entre referencia y retroalimentación
            'vd_correlation': np.corrcoef(self.df['vd_ref'], self.df['vd_act'])[0,1],
            'vi_correlation': np.corrcoef(self.df['vi_ref'], self.df['vi_act'])[0,1],
            
            # Tiempo de datos
            'sample_time': 0.048,  # 48ms basado en el log
            'total_time': len(self.df) * 0.048
        }
        
        print(f"✅ Métricas RMS calculadas para {len(self.df)} muestras")
    
    def create_dashboard(self):
        """Crear dashboard completo con análisis de lazos de control"""
        if self.df is None or len(self.df) == 0:
            print("❌ No hay datos para graficar")
            return
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('🤖 Robot Seguidor de Línea - Dashboard de Telemetría y Control', 
                     fontsize=18, fontweight='bold', color='white')
        
        # Layout de gráficas expandido
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. Lazo de velocidad derecha (Referencia vs Retroalimentación)
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_velocity_loop_tracking(ax1, 'right')
        
        # 2. Lazo de velocidad izquierda (Referencia vs Retroalimentación)
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_velocity_loop_tracking(ax2, 'left')
        
        # 3. Error de velocidad derecha en tiempo
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_velocity_error_time(ax3, 'right')
        
        # 4. Error de velocidad izquierda en tiempo
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_velocity_error_time(ax4, 'left')
        
        # 5. Análisis de correlación
        ax5 = fig.add_subplot(gs[1, 2])
        self.plot_correlation_analysis(ax5)
        
        # 6. Métricas RMS
        ax6 = fig.add_subplot(gs[1, 3])
        self.plot_rms_metrics(ax6)
        
        # 7. Posición de línea vs tiempo (con análisis)
        ax7 = fig.add_subplot(gs[2, :])
        self.plot_line_position_analysis(ax7)
        
        # 8. Trayectoria simulada
        ax8 = fig.add_subplot(gs[3, :2])
        self.plot_robot_trajectory(ax8)
        
        # 9. Estadísticas expandidas
        ax9 = fig.add_subplot(gs[3, 2:])
        self.plot_control_performance_stats(ax9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_velocity_loop_tracking(self, ax, side):
        """Gráfica de seguimiento de referencia para un lazo de velocidad"""
        time = self.df['t']
        
        if side == 'right':
            ref = self.df['vd_ref']
            act = self.df['vd_act']
            title = '🎯 Lazo Velocidad Derecha - Ref vs Retro'
            rms_error = self.rms_metrics['vd_error_rms']
            rms_percent = self.rms_metrics['vd_error_percent']
            correlation = self.rms_metrics['vd_correlation']
        else:
            ref = self.df['vi_ref']
            act = self.df['vi_act']
            title = '🎯 Lazo Velocidad Izquierda - Ref vs Retro'
            rms_error = self.rms_metrics['vi_error_rms']
            rms_percent = self.rms_metrics['vi_error_percent']
            correlation = self.rms_metrics['vi_correlation']
        
        # Plotear referencia y retroalimentación
        ax.plot(time, ref, color='#ef4444', linewidth=2, label='Referencia', alpha=0.9)
        ax.plot(time, act, color='#10b981', linewidth=2, label='Retroalimentación', alpha=0.9)
        
        # Área de error
        ax.fill_between(time, ref, act, alpha=0.2, color='#f59e0b', label='Error')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Velocidad')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Mostrar métricas en la gráfica
        metrics_text = f'RMS Error: {rms_error:.2f}\nError %: {rms_percent:.1f}%\nCorrelación: {correlation:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                fontsize=9, color='white')
    
    def plot_velocity_error_time(self, ax, side):
        """Gráfica del error de velocidad vs tiempo"""
        time = self.df['t']
        
        if side == 'right':
            error = self.df['vd_ref'] - self.df['vd_act']
            title = '📊 Error Velocidad Derecha'
            rms_error = self.rms_metrics['vd_error_rms']
        else:
            error = self.df['vi_ref'] - self.df['vi_act']
            title = '📊 Error Velocidad Izquierda'
            rms_error = self.rms_metrics['vi_error_rms']
        
        # Colorear según magnitud del error
        colors = ['#10b981' if abs(e) <= rms_error else '#f59e0b' if abs(e) <= 2*rms_error else '#ef4444' 
                 for e in error]
        
        ax.scatter(time, error, c=colors, s=1, alpha=0.7)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.axhline(y=rms_error, color='#f59e0b', linestyle=':', alpha=0.7, label=f'RMS: ±{rms_error:.2f}')
        ax.axhline(y=-rms_error, color='#f59e0b', linestyle=':', alpha=0.7)
        
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def plot_correlation_analysis(self, ax):
        """Análisis de correlación entre referencias y retroalimentación"""
        vd_corr = self.rms_metrics['vd_correlation']
        vi_corr = self.rms_metrics['vi_correlation']
        
        correlations = [vd_corr, vi_corr]
        labels = ['Vel Derecha', 'Vel Izquierda']
        colors = ['#ef4444', '#3b82f6']
        
        bars = ax.bar(labels, correlations, color=colors, alpha=0.8)
        
        # Línea de referencia para correlación perfecta
        ax.axhline(y=1.0, color='#10b981', linestyle='--', alpha=0.7, label='Correlación perfecta')
        ax.axhline(y=0.9, color='#f59e0b', linestyle=':', alpha=0.7, label='Correlación buena')
        
        ax.set_title('📈 Correlación Ref-Retro', fontweight='bold', fontsize=10)
        ax.set_ylabel('Correlación')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_rms_metrics(self, ax):
        """Mostrar métricas RMS de todos los lazos"""
        metrics = [
            ('VD Error RMS', self.rms_metrics['vd_error_rms'], '#ef4444'),
            ('VI Error RMS', self.rms_metrics['vi_error_rms'], '#3b82f6'),
            ('Pos Error RMS', self.rms_metrics['pos_error_rms'], '#8b5cf6')
        ]
        
        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        colors = [m[2] for m in metrics]
        
        bars = ax.bar(labels, values, color=colors, alpha=0.8)
        
        ax.set_title('🔢 Métricas RMS', fontweight='bold', fontsize=10)
        ax.set_ylabel('Valor RMS')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    def plot_line_position_analysis(self, ax):
        """Análisis expandido de posición de línea"""
        time = self.df['t']
        line_pos = self.df['line']
        
        # Plotear posición con banda de tolerancia
        ax.plot(time, line_pos, color='#3b82f6', linewidth=1, alpha=0.8, label='Posición línea')
        
        # Bandas de tolerancia
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.8, linewidth=2, label='Centro ideal')
        ax.fill_between(time, -20, 20, alpha=0.2, color='#10b981', label='Zona centrada (±20)')
        ax.fill_between(time, -50, -20, alpha=0.2, color='#f59e0b', label='Zona desviada')
        ax.fill_between(time, 20, 50, alpha=0.2, color='#f59e0b')
        ax.fill_between(time, -100, -50, alpha=0.2, color='#ef4444', label='Zona crítica')
        ax.fill_between(time, 50, 100, alpha=0.2, color='#ef4444')
        
        ax.set_title('📍 Análisis de Posición de Línea vs Tiempo', fontweight='bold')
        ax.set_xlabel('Tiempo (muestras)')
        ax.set_ylabel('Posición Línea')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Estadísticas de posición
        pos_rms = self.rms_metrics['pos_error_rms']
        stats_text = f'RMS Posición: {pos_rms:.1f}\nMax: {abs(line_pos).max():.0f}\nPromedio: {abs(line_pos).mean():.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                fontsize=9, color='white')
    
    def plot_robot_trajectory(self, ax):
        """Simular y graficar trayectoria del robot"""
        # Simulación simple de trayectoria basada en datos reales
        x, y = [0], [0]  # Posición inicial
        angle = 0        # Ángulo inicial
        
        for i in range(1, len(self.df)):
            dt = 0.048  # ~48ms entre muestras
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
        
        # Colorear trayectoria según calidad de control
        rms_threshold = self.rms_metrics['pos_error_rms']
        colors = ['#10b981' if abs(err) <= rms_threshold else '#f59e0b' if abs(err) <= 2*rms_threshold else '#ef4444' 
                 for err in self.df['line']]
        
        # Crear segmentos coloreados
        for i in range(len(x)-1):
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=2, alpha=0.8)
        
        # Marcar inicio y fin
        ax.plot(x[0], y[0], 'go', markersize=12, label='Inicio')
        ax.plot(x[-1], y[-1], 'ro', markersize=12, label='Fin')
        
        ax.set_title('🗺️ Trayectoria del Robot (Calidad de Control)', fontweight='bold')
        ax.set_xlabel('X (metros)')
        ax.set_ylabel('Y (metros)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Información de la trayectoria
        total_distance = sum(np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2) for i in range(len(x)-1))
        ax.text(0.02, 0.98, f'Distancia: {total_distance:.2f}m\nTiempo: {self.rms_metrics["total_time"]:.1f}s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                fontsize=9, color='white')
    
    def plot_control_performance_stats(self, ax):
        """Estadísticas expandidas de rendimiento de control"""
        ax.axis('off')
        ax.set_title('📊 Reporte de Rendimiento de Lazos de Control', fontweight='bold', pad=20, fontsize=12)
        
        # Organizar estadísticas en columnas
        col1_x, col2_x = 0.05, 0.55
        y_start = 0.95
        y_step = 0.08
        
        # Columna 1: Métricas RMS
        ax.text(col1_x, y_start, '🔢 MÉTRICAS RMS:', fontweight='bold', color='#3b82f6', 
               transform=ax.transAxes, fontsize=11)
        
        rms_stats = [
            f"Error VD: {self.rms_metrics['vd_error_rms']:.2f} ({self.rms_metrics['vd_error_percent']:.1f}%)",
            f"Error VI: {self.rms_metrics['vi_error_rms']:.2f} ({self.rms_metrics['vi_error_percent']:.1f}%)",
            f"Error Pos: {self.rms_metrics['pos_error_rms']:.2f}",
            f"Ref VD RMS: {self.rms_metrics['vd_ref_rms']:.2f}",
            f"Ref VI RMS: {self.rms_metrics['vi_ref_rms']:.2f}"
        ]
        
        for i, stat in enumerate(rms_stats):
            color = '#10b981' if 'Error' in stat and float(stat.split(':')[1].split()[0]) < 50 else '#f1f5f9'
            ax.text(col1_x, y_start - (i+1)*y_step, stat, color=color,
                   transform=ax.transAxes, fontsize=9)
        
        # Columna 2: Análisis de correlación y calidad
        ax.text(col2_x, y_start, '📈 CALIDAD DE CONTROL:', fontweight='bold', color='#8b5cf6',
               transform=ax.transAxes, fontsize=11)
        
        # Calcular estabilidad de línea
        line_stability = (abs(self.df['line']) <= 50).sum() / len(self.df) * 100
        max_line_error = abs(self.df['line']).max()
        
        quality_stats = [
            f"Correlación VD: {self.rms_metrics['vd_correlation']:.3f}",
            f"Correlación VI: {self.rms_metrics['vi_correlation']:.3f}",
            f"Estabilidad línea: {line_stability:.1f}%",
            f"Error máx línea: {max_line_error:.0f}",
            f"Tiempo total: {self.rms_metrics['total_time']:.1f}s"
        ]
        
        for i, stat in enumerate(quality_stats):
            if stat:
                if 'Correlación' in stat:
                    corr_val = float(stat.split(':')[1])
                    color = '#10b981' if corr_val > 0.7 else '#f59e0b' if corr_val > 0.6 else '#ef4444'
                elif 'Estabilidad' in stat:
                    stab_val = float(stat.split(':')[1].replace('%', ''))
                    color = '#10b981' if stab_val > 80 else '#f59e0b' if stab_val > 70 else '#ef4444'
                elif 'Error máx' in stat:
                    error_val = float(stat.split(':')[1])
                    color = '#10b981' if error_val < 100 else '#f59e0b' if error_val < 150 else '#ef4444'
                else:
                    color = '#f1f5f9'
                ax.text(col2_x, y_start - (i+1)*y_step, stat, color=color,
                       transform=ax.transAxes, fontsize=9)
        
        # Evaluación general
        ax.text(0.05, 0.25, '🎯 EVALUACIÓN GENERAL:', fontweight='bold', color='#ef4444',
               transform=ax.transAxes, fontsize=11)
        
        # Determinar calidad general del control (criterios realistas para seguidor de línea)
        avg_correlation = (self.rms_metrics['vd_correlation'] + self.rms_metrics['vi_correlation']) / 2
        avg_error_percent = (self.rms_metrics['vd_error_percent'] + self.rms_metrics['vi_error_percent']) / 2
        
        # Análisis adicional: estabilidad de seguimiento de línea
        line_stability = (abs(self.df['line']) <= 50).sum() / len(self.df) * 100  # % tiempo dentro de ±50
        max_line_error = abs(self.df['line']).max()
        
        # Criterios más realistas para robots seguidores de línea
        if avg_correlation > 0.85 and avg_error_percent < 15 and line_stability > 90:
            quality = "🟢 EXCELENTE - Control muy preciso y estable"
        elif avg_correlation > 0.7 and avg_error_percent < 25 and line_stability > 80:
            quality = "🟡 BUENO - Control funcional con oscilaciones normales"
        elif avg_correlation > 0.6 and avg_error_percent < 35 and line_stability > 70:
            quality = "🟠 REGULAR - Funciona pero puede mejorarse"
        elif avg_correlation > 0.5 and max_line_error < 100 and line_stability > 60:
            quality = "🟡 ACEPTABLE - Robot no se sale, oscilaciones controladas"
        else:
            quality = "🔴 DEFICIENTE - Riesgo de perder línea"
        
        ax.text(0.05, 0.15, quality, fontweight='bold', color='white',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    def create_analysis_report(self):
        """Crear reporte expandido de análisis del comportamiento"""
        if self.df is None:
            return
        
        print("\n🤖 === REPORTE EXPANDIDO DE ANÁLISIS DEL ROBOT ===")
        print("=" * 60)
        
        # Análisis de lazos de velocidad
        print(f"🎯 ANÁLISIS DE LAZOS DE VELOCIDAD:")
        print(f"   Lazo Velocidad Derecha:")
        print(f"     • RMS Error: {self.rms_metrics['vd_error_rms']:.2f}")
        print(f"     • Error porcentual: {self.rms_metrics['vd_error_percent']:.1f}%")
        print(f"     • Correlación Ref-Retro: {self.rms_metrics['vd_correlation']:.3f}")
        print(f"     • RMS Referencia: {self.rms_metrics['vd_ref_rms']:.2f}")
        print(f"     • RMS Retroalimentación: {self.rms_metrics['vd_act_rms']:.2f}")
        
        print(f"\n   Lazo Velocidad Izquierda:")
        print(f"     • RMS Error: {self.rms_metrics['vi_error_rms']:.2f}")
        print(f"     • Error porcentual: {self.rms_metrics['vi_error_percent']:.1f}%")
        print(f"     • Correlación Ref-Retro: {self.rms_metrics['vi_correlation']:.3f}")
        print(f"     • RMS Referencia: {self.rms_metrics['vi_ref_rms']:.2f}")
        print(f"     • RMS Retroalimentación: {self.rms_metrics['vi_act_rms']:.2f}")
        
        # Análisis de posición
        print(f"\n📍 ANÁLISIS DE CONTROL DE POSICIÓN:")
        print(f"   • RMS Error Posición: {self.rms_metrics['pos_error_rms']:.2f}")
        print(f"   • Error promedio: {abs(self.df['pos_err']).mean():.1f}")
        print(f"   • Error máximo: {abs(self.df['pos_err']).max():.0f}")
        
        # Análisis de seguimiento de línea
        total_samples = len(self.df)
        centered_samples = (abs(self.df['line']) <= 20).sum()
        deviated_samples = (abs(self.df['line']) > 50).sum()
        
        print(f"\n📊 RENDIMIENTO DE SEGUIMIENTO:")
        print(f"   • Tiempo total: {self.rms_metrics['total_time']:.1f} segundos")
        print(f"   • Frecuencia de muestreo: {1/self.rms_metrics['sample_time']:.1f} Hz")
        print(f"   • Tiempo centrado (±20): {centered_samples/total_samples*100:.1f}%")
        print(f"   • Tiempo desviado (>50): {deviated_samples/total_samples*100:.1f}%")
        
        # Evaluación de calidad de control
        print(f"\n🎯 EVALUACIÓN DE CALIDAD DE CONTROL:")
        
        # Criterios de evaluación más realistas
        vd_quality = "EXCELENTE" if self.rms_metrics['vd_correlation'] > 0.85 and self.rms_metrics['vd_error_percent'] < 15 else \
                    "BUENO" if self.rms_metrics['vd_correlation'] > 0.7 and self.rms_metrics['vd_error_percent'] < 25 else \
                    "ACEPTABLE" if self.rms_metrics['vd_correlation'] > 0.6 and self.rms_metrics['vd_error_percent'] < 35 else \
                    "REGULAR" if self.rms_metrics['vd_correlation'] > 0.5 else "DEFICIENTE"
        
        vi_quality = "EXCELENTE" if self.rms_metrics['vi_correlation'] > 0.85 and self.rms_metrics['vi_error_percent'] < 15 else \
                    "BUENO" if self.rms_metrics['vi_correlation'] > 0.7 and self.rms_metrics['vi_error_percent'] < 25 else \
                    "ACEPTABLE" if self.rms_metrics['vi_correlation'] > 0.6 and self.rms_metrics['vi_error_percent'] < 35 else \
                    "REGULAR" if self.rms_metrics['vi_correlation'] > 0.5 else "DEFICIENTE"
        
        print(f"   • Lazo Velocidad Derecha: {vd_quality}")
        print(f"   • Lazo Velocidad Izquierda: {vi_quality}")
        
        # Recomendaciones más realistas
        print(f"\n💡 RECOMENDACIONES:")
        if self.rms_metrics['vd_error_percent'] > 30:
            print(f"   ⚠️  Considerar ajustar ganancias del controlador de velocidad derecha")
        if self.rms_metrics['vi_error_percent'] > 30:
            print(f"   ⚠️  Considerar ajustar ganancias del controlador de velocidad izquierda")
        if self.rms_metrics['vd_correlation'] < 0.6 or self.rms_metrics['vi_correlation'] < 0.6:
            print(f"   ⚠️  Verificar sintonización de controladores PID")
        if deviated_samples > total_samples * 0.3:
            print(f"   ⚠️  Revisar algoritmo de detección de línea")
        if abs(self.df['line']).max() > 100:
            print(f"   ⚠️  Robot cerca del límite - revisar parámetros de velocidad")
        
        # Evaluación positiva si funciona bien
        line_stability = (abs(self.df['line']) <= 50).sum() / len(self.df) * 100
        if line_stability > 80 and abs(self.df['line']).max() < 100:
            print(f"   ✅ Robot mantiene buen seguimiento de línea")
        if self.rms_metrics['vd_correlation'] > 0.7 and self.rms_metrics['vi_correlation'] > 0.7:
            print(f"   ✅ Lazos de velocidad responden adecuadamente")
        
        print("=" * 60)
    
    def run_analysis(self):
        """Ejecutar análisis completo con métricas RMS"""
        if not self.load_data():
            return
        
        self.create_analysis_report()
        print(f"\n📊 Generando dashboard con análisis RMS...")
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