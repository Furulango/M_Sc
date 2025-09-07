import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.integrate import solve_ivp
import ast # Para convertir strings de listas a listas

# ===============================================================================
# MARCO DE SIMULACIÓN DE MOTOR (Extraído de tu script)
# ===============================================================================
def induction_motor(t, x, params, vqs, vds):
    """Ecuaciones diferenciales del motor de inducción en el marco de referencia dq."""
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2 * np.pi * 60
    ws = we - wr
    lqs = Ls * iqs + Lm * iqr
    lds = Ls * ids + Lm * idr
    lqr = Lr * iqr + Lm * iqs
    ldr = Lr * idr + Lm * ids
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs * iqs - we * lds, vds - rs * ids + we * lqs, -rr * iqr - ws * ldr, -rr * idr + ws * lqr])
    di_dt = np.linalg.solve(L, v)
    Te = (3 * 4 / 4) * Lm * (iqs * idr - ids * iqr)
    dwr_dt = (Te - B * wr) / J
    return np.array([*di_dt, dwr_dt])

def simulate_motor(params, t_span=[0, 2], n_points=400):
    """Ejecuta una simulación del motor con un conjunto de parámetros dado."""
    vqs, vds = 220 * np.sqrt(2) / np.sqrt(3), 0
    try:
        sol = solve_ivp(lambda t, x: induction_motor(t, x, params, vqs, vds),
                        t_span, [0, 0, 0, 0, 0], dense_output=True, rtol=1e-6, atol=1e-8)
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        Is_mag = np.sqrt(iqs**2 + ids**2)
        Te = (3 * 4 / 4) * params[4] * (iqs * idr - ids * iqr)
        rpm = wr * 60 / (2 * np.pi) * 2 / 4
        return t, {'Is_mag': Is_mag, 'Te': Te, 'rpm': rpm}
    except Exception:
        # En caso de fallo en la simulación, devuelve valores altos para penalizar
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'Is_mag': np.full(n_points, np.nan),
                   'Te': np.full(n_points, np.nan),
                   'rpm': np.full(n_points, np.nan)}

# ===============================================================================
# FUNCIÓN DE VISUALIZACIÓN DE ERRORES (Boxplots) - MODIFICADA PARA GUARDAR
# ===============================================================================
def plot_error_distribution(combined_df, algorithms, save_path='./graficas/'):
    """
    Genera boxplots para comparar el error en la identificación de parámetros.
    MODIFICADO: Título principal mejorado con escenario más prominente y GUARDA las gráficas.
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    print("--- Generando y Guardando Gráficas de Distribución de Error (Boxplots) ---")
    error_columns = [col for col in combined_df.columns if col.startswith('error_')]
    scenarios = combined_df['scenario'].unique()

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", n_colors=len(algorithms))

    for scenario in scenarios:
        print(f"Generando boxplots para el escenario: {scenario}")
        scenario_df = combined_df[combined_df['scenario'] == scenario]
        n_plots = len(error_columns)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        # --- TÍTULO PRINCIPAL MEJORADO CON ESCENARIO MÁS PROMINENTE ---
        fig.suptitle(f'ESCENARIO: {scenario}\nDistribución de Error por Parámetro', 
                    fontsize=22, fontweight='bold', y=1.02)

        for i, err_col in enumerate(error_columns):
            ax = axes[i]
            sns.boxplot(x='algorithm', y=err_col, data=scenario_df, ax=ax, palette=palette, showfliers=False)
            param_name = err_col.replace('error_', '')
            ax.set_title(f'Error en {param_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Algoritmo', fontweight='bold')
            ax.set_ylabel('Error (%)', fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            
            # Agregar estadísticas básicas en cada subplot
            medians = scenario_df.groupby('algorithm')[err_col].median()
            for j, (algo, median_val) in enumerate(medians.items()):
                ax.text(j, ax.get_ylim()[1] * 0.95, f'Med: {median_val:.2f}%', 
                       horizontalalignment='center', fontsize=10, fontweight='bold')

        # Ocultar subplots vacíos
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # GUARDAR LA GRÁFICA
        filename = f"Boxplots_Error_Escenario_{scenario}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Guardado: {filepath}")
        
        plt.show()
        plt.close()  # Liberar memoria

# ===============================================================================
# FUNCIÓN DE VISUALIZACIÓN DE SIMULACIONES (Curvas de Tiempo) - MEJORADA PARA GUARDAR
# ===============================================================================
def plot_simulation_comparison(combined_df, algorithms, save_path='./graficas/'):
    """
    Genera gráficas comparando las simulaciones del motor (corriente, torque, rpm)
    usando los mejores parámetros identificados por cada algoritmo contra los reales.
    MEJORADO: Título con escenario más prominente y GUARDA las gráficas.
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    print("\n--- Generando y Guardando Gráficas de Comparación de Simulaciones ---")
    scenarios = combined_df['scenario'].unique()
    sns.set_theme(style="darkgrid")
    
    for scenario in scenarios:
        print(f"Simulando para el escenario: {scenario}")
        scenario_df = combined_df[combined_df['scenario'] == scenario].copy()
        
        # Obtener los parámetros reales (ground truth) para este escenario
        try:
            true_params_str = scenario_df['true_params'].iloc[0]
            true_params = np.array(ast.literal_eval(true_params_str))
        except (ValueError, SyntaxError) as e:
            print(f"Error al procesar 'true_params' en el escenario {scenario}: {e}")
            continue

        # Simular el comportamiento real
        t_real, real_outputs = simulate_motor(true_params)

        # Preparar la figura para las 3 gráficas (Corriente, Torque, RPM)
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
        
        # --- TÍTULO PRINCIPAL MEJORADO CON ESCENARIO MÁS PROMINENTE ---
        fig.suptitle(f'ESCENARIO: {scenario}\nComparación de Simulación vs. Real', 
                    fontsize=24, fontweight='bold', y=0.96)

        # Graficar la curva real en cada subplot
        axes[0].plot(t_real, real_outputs['Is_mag'], 'k--', linewidth=3, label='Real', alpha=0.8)
        axes[1].plot(t_real, real_outputs['Te'], 'k--', linewidth=3, label='Real', alpha=0.8)
        axes[2].plot(t_real, real_outputs['rpm'], 'k--', linewidth=3, label='Real', alpha=0.8)

        # Iterar sobre cada algoritmo para encontrar su mejor resultado y simularlo
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for idx, algo_name in enumerate(algorithms):
            algo_df = scenario_df[scenario_df['algorithm'] == algo_name]
            if algo_df.empty:
                continue

            # Encontrar la mejor ejecución (menor error)
            best_run = algo_df.loc[algo_df['error'].idxmin()]
            
            try:
                identified_params = np.array(ast.literal_eval(best_run['identified_params']))
            except (ValueError, SyntaxError) as e:
                print(f"Error al procesar 'identified_params' para {algo_name} en {scenario}: {e}")
                continue

            # Simular con los parámetros identificados
            t_sim, sim_outputs = simulate_motor(identified_params)
            
            color = colors[idx % len(colors)]
            
            # Graficar los resultados de la simulación
            axes[0].plot(t_sim, sim_outputs['Is_mag'], 
                        label=f'{algo_name} (Error: {best_run["error"]:.2f}%)', 
                        alpha=0.8, linewidth=2, color=color)
            axes[1].plot(t_sim, sim_outputs['Te'], 
                        label=f'{algo_name}', 
                        alpha=0.8, linewidth=2, color=color)
            axes[2].plot(t_sim, sim_outputs['rpm'], 
                        label=f'{algo_name}', 
                        alpha=0.8, linewidth=2, color=color)

        # Configurar títulos y etiquetas con mejor formato
        axes[0].set_title('Corriente del Estator (Is)', fontsize=18, fontweight='bold', pad=20)
        axes[0].set_ylabel('Corriente (A)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=12, loc='upper right')
        axes[0].grid(True, which='both', linestyle=':', alpha=0.7)

        axes[1].set_title('Torque Electromagnético (Te)', fontsize=18, fontweight='bold', pad=20)
        axes[1].set_ylabel('Torque (Nm)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=12, loc='upper right')
        axes[1].grid(True, which='both', linestyle=':', alpha=0.7)

        axes[2].set_title('Velocidad del Motor', fontsize=18, fontweight='bold', pad=20)
        axes[2].set_ylabel('Velocidad (RPM)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Tiempo (s)', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=12, loc='upper right')
        axes[2].grid(True, which='both', linestyle=':', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        # GUARDAR LA GRÁFICA
        filename = f"Simulacion_Comparativa_Escenario_{scenario}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📈 Guardado: {filepath}")
        
        plt.show()
        plt.close()  # Liberar memoria

# ===============================================================================
# FUNCIÓN ADICIONAL: RESUMEN ESTADÍSTICO POR ESCENARIO Y GUARDAR REPORTE
# ===============================================================================
def print_scenario_summary(combined_df, algorithms, save_path='./graficas/'):
    """
    Imprime un resumen estadístico por escenario y algoritmo y lo GUARDA en archivo.
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*80)
    print("RESUMEN ESTADÍSTICO POR ESCENARIO")
    print("="*80)
    
    # Preparar contenido para guardar
    report_content = []
    report_content.append("="*80)
    report_content.append("REPORTE ESTADÍSTICO - ANÁLISIS DE ALGORITMOS DE OPTIMIZACIÓN")
    report_content.append("="*80)
    report_content.append(f"Fecha de generación: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Algoritmos analizados: {', '.join(algorithms)}")
    report_content.append("")
    
    scenarios = combined_df['scenario'].unique()
    
    for scenario in scenarios:
        print(f"\n--- ESCENARIO: {scenario} ---")
        scenario_df = combined_df[combined_df['scenario'] == scenario]
        
        summary_stats = scenario_df.groupby('algorithm')['error'].agg(['mean', 'std', 'min', 'max', 'median'])
        print(summary_stats.round(3))
        
        # Agregar al reporte
        report_content.append(f"--- ESCENARIO: {scenario} ---")
        report_content.append("Estadísticas de Error Global por Algoritmo:")
        report_content.append(summary_stats.round(3).to_string())
        
        # Mejor algoritmo para este escenario
        best_algo = summary_stats['median'].idxmin()
        best_median = summary_stats['median'].min()
        print(f"\nMejor algoritmo (por mediana): {best_algo} con {best_median:.3f}% de error")
        
        report_content.append(f"\nMejor algoritmo (por mediana): {best_algo} con {best_median:.3f}% de error")
        
        # Análisis detallado por parámetro
        error_columns = [col for col in combined_df.columns if col.startswith('error_')]
        report_content.append("\nAnálisis detallado por parámetro:")
        for err_col in error_columns:
            param_name = err_col.replace('error_', '')
            param_stats = scenario_df.groupby('algorithm')[err_col].agg(['mean', 'median'])
            best_param_algo = param_stats['median'].idxmin()
            best_param_error = param_stats['median'].min()
            report_content.append(f"  {param_name}: Mejor = {best_param_algo} ({best_param_error:.3f}%)")
        
        print("-" * 50)
        report_content.append("-" * 50)
        report_content.append("")
    
    # Análisis general
    print(f"\n{'='*30} ANÁLISIS GENERAL {'='*30}")
    report_content.append(f"{'='*30} ANÁLISIS GENERAL {'='*30}")
    
    overall_stats = combined_df.groupby('algorithm')['error'].agg(['mean', 'std', 'median', 'count'])
    print("Estadísticas generales (todos los escenarios):")
    print(overall_stats.round(3))
    
    report_content.append("Estadísticas generales (todos los escenarios):")
    report_content.append(overall_stats.round(3).to_string())
    
    best_overall = overall_stats['median'].idxmin()
    print(f"\nMejor algoritmo general: {best_overall}")
    report_content.append(f"\nMejor algoritmo general: {best_overall}")
    
    # GUARDAR EL REPORTE
    report_filename = os.path.join(save_path, "Reporte_Estadistico_Completo.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    print(f"\n📋 Reporte guardado: {report_filename}")

# ===============================================================================
# FUNCIÓN PARA CREAR REPORTE PDF (OPCIONAL)
# ===============================================================================
def create_summary_plot(combined_df, algorithms, save_path='./graficas/'):
    """
    Crea una gráfica resumen con todos los algoritmos y escenarios.
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(16, 10))
    
    # Gráfica de medianas por escenario y algoritmo
    pivot_data = combined_df.pivot_table(values='error', index='scenario', columns='algorithm', aggfunc='median')
    
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Error Mediano (%)'})
    
    plt.title('Resumen: Error Mediano por Escenario y Algoritmo', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Algoritmo', fontsize=14, fontweight='bold')
    plt.ylabel('Escenario', fontsize=14, fontweight='bold')
    
    # Guardar
    filename = os.path.join(save_path, "Resumen_Heatmap_Errores.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🔥 Mapa de calor guardado: {filename}")
    
    plt.show()
    plt.close()

# --- Ejecución del script ---
if __name__ == "__main__":
    # Configuración de rutas
    SAVE_PATH = './graficas_resultados/'  # Carpeta donde se guardarán las gráficas
    
    # Nombres de los archivos que subiste y los algoritmos correspondientes
    file_paths = [
        'BFO_adaptive_study.csv',
        'PSO_adaptive_study.csv',
        'Chaotic_PSODSO_adaptive_study.csv'
    ]
    algorithm_names = [
        'BFO',
        'PSO',
        'Chaotic PSODSO'
    ]

    print("🚀 INICIANDO ANÁLISIS Y GENERACIÓN DE GRÁFICAS")
    print("="*60)

    # Cargar y combinar los dataframes
    all_data = []
    for file_path, algo_name in zip(file_paths, algorithm_names):
        try:
            df = pd.read_csv(file_path)
            df['algorithm'] = algo_name
            all_data.append(df)
            print(f"✅ Cargado exitosamente: {file_path}")
        except FileNotFoundError:
            print(f"⚠️  Advertencia: El archivo {file_path} no fue encontrado y será omitido.")
    
    if not all_data:
        print("❌ Error: No se pudieron cargar datos. Verifica las rutas de los archivos.")
    else:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Total de registros cargados: {len(combined_df)}")
        print(f"📋 Escenarios encontrados: {list(combined_df['scenario'].unique())}")
        print(f"🔬 Algoritmos encontrados: {list(combined_df['algorithm'].unique())}")
        print(f"💾 Las gráficas se guardarán en: {SAVE_PATH}")

        # Crear directorio de salida
        import os
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {SAVE_PATH}")

        print("\n" + "="*60)
        print("GENERANDO ANÁLISIS COMPLETO...")
        print("="*60)

        # 1. Mostrar resumen estadístico y guardarlo
        print_scenario_summary(combined_df, algorithm_names, SAVE_PATH)

        # 2. Generar las gráficas de distribución de error y guardarlas
        plot_error_distribution(combined_df, algorithm_names, SAVE_PATH)

        # 3. Generar las gráficas de simulación y guardarlas
        plot_simulation_comparison(combined_df, algorithm_names, SAVE_PATH)

        # 4. Crear gráfica resumen (heatmap)
        create_summary_plot(combined_df, algorithm_names, SAVE_PATH)

        print("\n" + "="*60)
        print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        print(f"📂 Todas las gráficas y reportes se han guardado en: {SAVE_PATH}")
        
        # Listar archivos generados
        import os
        generated_files = [f for f in os.listdir(SAVE_PATH) if f.endswith(('.png', '.txt'))]
        print(f"📋 Archivos generados ({len(generated_files)}):")
        for file in sorted(generated_files):
            print(f"   📄 {file}")
        