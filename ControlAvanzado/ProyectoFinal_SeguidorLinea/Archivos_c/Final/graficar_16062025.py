import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Cargar los datos del JSON
def load_data(filename):
    """Carga los datos del archivo JSON línea por línea"""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# Cargar datos
df = load_data('datos_velocidades_reales.json')

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Seguimiento de Posición
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(df['t'], df['pos_ref'], 'b-', label='Posición Referencia', linewidth=2)
ax1.plot(df['t'], df['pos_act'], 'r--', label='Posición Actual', linewidth=2)
ax1.fill_between(df['t'], df['pos_ref'], df['pos_act'], alpha=0.3, color='gray')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Posición')
ax1.set_title('Seguimiento de Posición del Seguidor de Línea')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error de Posición
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(df['t'], df['pos_err'], 'g-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Error de Posición')
ax2.set_title('Error de Posición')
ax2.grid(True, alpha=0.3)
# Resaltar zonas de mayor error
ax2.fill_between(df['t'], df['pos_err'], 0, where=(abs(df['pos_err']) > 100), 
                 color='red', alpha=0.3, label='Error > 100')
ax2.legend()

# 3. Velocidades de Rueda Derecha
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(df['t'], df['vel_d_ref'], 'b-', label='Vel. Der. Ref.', linewidth=2)
ax3.plot(df['t'], df['vel_d_act'], 'r--', label='Vel. Der. Actual', linewidth=2)
ax3.set_xlabel('Tiempo')
ax3.set_ylabel('Velocidad Derecha')
ax3.set_title('Control de Velocidad - Rueda Derecha')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Velocidades de Rueda Izquierda
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df['t'], df['vel_i_ref'], 'b-', label='Vel. Izq. Ref.', linewidth=2)
ax4.plot(df['t'], df['vel_i_act'], 'r--', label='Vel. Izq. Actual', linewidth=2)
ax4.set_xlabel('Tiempo')
ax4.set_ylabel('Velocidad Izquierda')
ax4.set_title('Control de Velocidad - Rueda Izquierda')
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

# 6. Salida del Controlador de Posición
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(df['t'], df['pos_out'], 'purple', linewidth=2)
ax6.set_xlabel('Tiempo')
ax6.set_ylabel('Salida del Controlador')
ax6.set_title('Señal de Control de Posición')
ax6.grid(True, alpha=0.3)

# 7. Análisis de Correlación Velocidades
ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(df['vel_d_ref'], df['vel_d_act'], alpha=0.6, color='red', label='Rueda Der.')
ax7.scatter(df['vel_i_ref'], df['vel_i_act'], alpha=0.6, color='blue', label='Rueda Izq.')
# Línea perfecta de seguimiento
min_vel = min(df[['vel_d_ref', 'vel_i_ref', 'vel_d_act', 'vel_i_act']].min())
max_vel = max(df[['vel_d_ref', 'vel_i_ref', 'vel_d_act', 'vel_i_act']].max())
ax7.plot([min_vel, max_vel], [min_vel, max_vel], 'k--', alpha=0.8, label='Seguimiento Perfecto')
ax7.set_xlabel('Velocidad Referencia')
ax7.set_ylabel('Velocidad Actual')
ax7.set_title('Correlación Referencia vs Actual')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Histograma de Errores
ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(df['pos_err'], bins=30, alpha=0.7, color='green', label='Error Posición')
ax8.axvline(df['pos_err'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["pos_err"].mean():.1f}')
ax8.set_xlabel('Error de Posición')
ax8.set_ylabel('Frecuencia')
ax8.set_title('Distribución del Error de Posición')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Métricas de Rendimiento (Tabla)
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

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

# Crear tabla de métricas
table_data = []
for control_type, metrics_dict in metrics.items():
    for metric_name, value in metrics_dict.items():
        table_data.append([control_type, metric_name, f'{value:.2f}'])

table = ax9.table(cellText=table_data,
                  colLabels=['Sistema', 'Métrica', 'Valor'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.3, 0.3, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax9.set_title('Métricas de Rendimiento del Sistema', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Análisis Completo del Seguidor de Línea', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Análisis adicional y recomendaciones
print("="*60)
print("ANÁLISIS DEL RENDIMIENTO DEL SEGUIDOR DE LÍNEA")
print("="*60)

print(f"\n📊 RESUMEN ESTADÍSTICO:")
print(f"Tiempo de simulación: {df['t'].iloc[-1]} unidades")
print(f"Total de muestras: {len(df)}")

print(f"\n🎯 CONTROL DE POSICIÓN:")
print(f"  • Error RMS: {metrics['Posición']['RMSE']:.2f}")
print(f"  • Error promedio absoluto: {metrics['Posición']['MAE']:.2f}")
print(f"  • Error máximo: {metrics['Posición']['Error Max']:.2f}")
print(f"  • Desviación estándar: {metrics['Posición']['Desv. Std']:.2f}")

print(f"\n⚙️ CONTROL DE VELOCIDADES:")
print(f"  Rueda Derecha:")
print(f"    • Error RMS: {metrics['Vel. Derecha']['RMSE']:.2f}")
print(f"    • Error máximo: {metrics['Vel. Derecha']['Error Max']:.2f}")
print(f"  Rueda Izquierda:")
print(f"    • Error RMS: {metrics['Vel. Izquierda']['RMSE']:.2f}")
print(f"    • Error máximo: {metrics['Vel. Izquierda']['Error Max']:.2f}")

# Evaluación del rendimiento
pos_performance = "EXCELENTE" if metrics['Posición']['RMSE'] < 50 else "BUENO" if metrics['Posición']['RMSE'] < 100 else "REGULAR"
vel_performance = "EXCELENTE" if max(metrics['Vel. Derecha']['RMSE'], metrics['Vel. Izquierda']['RMSE']) < 1000 else "BUENO" if max(metrics['Vel. Derecha']['RMSE'], metrics['Vel. Izquierda']['RMSE']) < 2000 else "REGULAR"

print(f"\n✅ EVALUACIÓN DEL RENDIMIENTO:")
print(f"  • Control de Posición: {pos_performance}")
print(f"  • Control de Velocidades: {vel_performance}")

print(f"\n💡 RECOMENDACIONES:")
if metrics['Posición']['RMSE'] > 100:
    print("  • Considerar ajustar las ganancias del controlador de posición")
if max(metrics['Vel. Derecha']['RMSE'], metrics['Vel. Izquierda']['RMSE']) > 2000:
    print("  • Revisar la calibración de los motores")
if abs(metrics['Vel. Derecha']['RMSE'] - metrics['Vel. Izquierda']['RMSE']) > 500:
    print("  • Existe desbalance entre las ruedas, verificar mecánica")

print(f"\n📈 ANÁLISIS DE ESTABILIDAD:")
pos_settling = len(df[abs(df['pos_err']) > 50]) / len(df) * 100
print(f"  • Tiempo fuera de banda (±50): {pos_settling:.1f}%")
if pos_settling < 20:
    print("  • Sistema ESTABLE ✅")
elif pos_settling < 40:
    print("  • Sistema MODERADAMENTE ESTABLE ⚠️")
else:
    print("  • Sistema INESTABLE ❌")