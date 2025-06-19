import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Cargar los datos JSON
def load_data(filename):
    """Carga los datos del archivo JSON línea por línea"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# Cargar datos
df = load_data('line_seeker_2025_v1.json')

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

# Crear figura con múltiples subplots - 2 filas, 3 columnas
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Seguimiento Posición
ax1 = fig.add_subplot(gs[0, :2])

# *** ZONAS DE RIESGO AGREGADAS ***
# Zona Crítica: ±300 a ±400
ax1.axhspan(-400, -300, alpha=0.3, color='red', label=' Zona Crítica (±300-400)')
ax1.axhspan(300, 400, alpha=0.3, color='red')

# Zona Advertencia: ±150 a ±300  
ax1.axhspan(-300, -150, alpha=0.25, color='yellow', label=' Zona Advertencia (±150-300)')
ax1.axhspan(150, 300, alpha=0.25, color='yellow')

# Zona Control Óptimo: ±150
ax1.axhspan(-150, 150, alpha=0.15, color='lightgreen', label=' Zona Control Óptimo (±150)')

# Líneas de referencia
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Centro de Línea')
ax1.axhline(y=150, color='green', linestyle='--', alpha=0.6, linewidth=1)
ax1.axhline(y=-150, color='green', linestyle='--', alpha=0.6, linewidth=1)
ax1.axhline(y=300, color='orange', linestyle='--', alpha=0.7, linewidth=1)
ax1.axhline(y=-300, color='orange', linestyle='--', alpha=0.7, linewidth=1)

# Posición
ax1.plot(df['t'], df['pos_act'], 'red', label='Posición del Robot', linewidth=1.5)
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Posición Lateral')
ax1.set_title('Control de Seguimiento de Línea - Desempeño con Zonas de Riesgo')
ax1.set_ylim(-400, 400)  # Ampliado para mostrar zona crítica
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Estadísticas
tiempo_precision = np.sum(np.abs(df['pos_act']) <= 50) / len(df) * 100
tiempo_control = np.sum(np.abs(df['pos_act']) <= 150) / len(df) * 100
tiempo_advertencia = np.sum((np.abs(df['pos_act']) > 150) & (np.abs(df['pos_act']) <= 300)) / len(df) * 100
tiempo_critico = np.sum(np.abs(df['pos_act']) > 300) / len(df) * 100

ax1.text(0.02, 0.98, f'🟢 Zona Óptima (±150): {tiempo_control:.1f}%\n🟡 Zona Advertencia: {tiempo_advertencia:.1f}%\n🔴 Zona Crítica (>300): {tiempo_critico:.1f}%\n⭐ Alta Precisión (±50): {tiempo_precision:.1f}%', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

# Histograma de Errores
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df['pos_err'], bins=30, alpha=0.7, color='green', label='Error Posición')
ax2.axvline(df['pos_err'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["pos_err"].mean():.1f}')
# Agregar líneas de zonas en histograma
ax2.axvline(150, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Límite Advertencia')
ax2.axvline(-150, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax2.axvline(300, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Límite Crítico')
ax2.axvline(-300, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax2.set_xlabel('Error de Posición')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Histograma de Errores de Posición')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Velocidades de Motor Derecha
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df['t'], df['vel_d_ref'], 'b-', label='Vel. Der. Ref.', linewidth=2)
ax3.plot(df['t'], df['vel_d_act'], 'r--', label='Vel. Der. Actual', linewidth=2)
ax3.set_xlabel('Tiempo')
ax3.set_ylabel('Velocidad Derecha')
ax3.set_title('Velocidades de Motor Derecha')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Velocidades de Motor Izquierda
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df['t'], df['vel_i_ref'], 'b-', label='Vel. Izq. Ref.', linewidth=2)
ax4.plot(df['t'], df['vel_i_act'], 'r--', label='Vel. Izq. Actual', linewidth=2)
ax4.set_xlabel('Tiempo')
ax4.set_ylabel('Velocidad Izquierda')
ax4.set_title('Velocidades de Motor Izquierda')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Errores de Velocidad
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(df['t'], df['vel_d_err'], 'r-', label='Error Vel. Der.', linewidth=2)
ax5.plot(df['t'], df['vel_i_err'], 'b-', label='Error Vel. Izq.', linewidth=2)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax5.set_xlabel('Tiempo')
ax5.set_ylabel('Error de Velocidad')
ax5.set_title('Errores de Velocidad')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.suptitle('Análisis de Seguidor de Línea con Zonas de Riesgo', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("="*80)
print(" ANÁLISIS")
print("="*80)

print(f"\n RESUMEN:")
print(f"Tiempo de simulación: {df['t'].iloc[-1]} unidades")
print(f"Total de muestras: {len(df)}")

# Calcular métricas
metrics = {
    'Posición': {
        'RMSE': np.sqrt(np.mean(df['pos_err']**2)),
        'MAE': np.mean(np.abs(df['pos_err'])),
        'Error Max': np.max(np.abs(df['pos_err'])),
        'Desv. Std': np.std(df['pos_err'])
    },
    'Vel. Derecha': {
        'RMSE': np.sqrt(np.mean(df['vel_d_err']**2)),
        'MAE': np.mean(np.abs(df['vel_d_err'])),
        'Error Max': np.max(np.abs(df['vel_d_err'])),
        'Desv. Std': np.std(df['vel_d_err'])
    },
    'Vel. Izquierda': {
        'RMSE': np.sqrt(np.mean(df['vel_i_err']**2)),
        'MAE': np.mean(np.abs(df['vel_i_err'])),
        'Error Max': np.max(np.abs(df['vel_i_err'])),
        'Desv. Std': np.std(df['vel_i_err'])
    }
}

print(f"\n CONTROL DE POSICIÓN:")
print(f"  • Error RMS: {metrics['Posición']['RMSE']:.2f}")
print(f"  • Error promedio absoluto: {metrics['Posición']['MAE']:.2f}")
print(f"  • Error máximo: {metrics['Posición']['Error Max']:.2f}")
print(f"  • Desviación estándar: {metrics['Posición']['Desv. Std']:.2f}")

print(f"\n DISTRIBUCIÓN POR ZONAS DE RIESGO:")
print(f"   Zona Óptima (±150): {tiempo_control:.1f}% del tiempo")
print(f"   Zona Advertencia (150-300): {tiempo_advertencia:.1f}% del tiempo")
print(f"   Zona Crítica (>300): {tiempo_critico:.1f}% del tiempo")
print(f"   Alta Precisión (±50): {tiempo_precision:.1f}% del tiempo")

print(f"\n CONTROL DE VELOCIDADES:")
print(f"  Rueda Derecha:")
print(f"    • Error RMS: {metrics['Vel. Derecha']['RMSE']:.2f}")
print(f"    • Error máximo: {metrics['Vel. Derecha']['Error Max']:.2f}")
print(f"  Rueda Izquierda:")
print(f"    • Error RMS: {metrics['Vel. Izquierda']['RMSE']:.2f}")
print(f"    • Error máximo: {metrics['Vel. Izquierda']['Error Max']:.2f}")
