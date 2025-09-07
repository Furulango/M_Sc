#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS - DIGITAL TWIN ADAPTATIVO
Genera gráficas detalladas de los resultados de optimización
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Crear directorio para gráficas
os.makedirs('analysis_plots', exist_ok=True)

# ===============================================================================
# FUNCIONES DEL MODELO DEL MOTOR (copiadas del código original)
# ===============================================================================

def induction_motor(t, x, params, vqs, vds):
    """Modelo dinámico del motor de inducción"""
    iqs, ids, iqr, idr, wr = x
    rs, rr, Lls, Llr, Lm, J, B = params
    Ls, Lr = Lls + Lm, Llr + Lm
    we = 2*np.pi*60
    ws = we - wr
    lqs = Ls*iqs + Lm*iqr
    lds = Ls*ids + Lm*idr
    lqr = Lr*iqr + Lm*iqs
    ldr = Lr*idr + Lm*ids
    L = np.array([[Ls, 0, Lm, 0], [0, Ls, 0, Lm], [Lm, 0, Lr, 0], [0, Lm, 0, Lr]])
    v = np.array([vqs - rs*iqs - we*lds, vds - rs*ids + we*lqs, -rr*iqr - ws*ldr, -rr*idr + ws*lqr])
    di_dt = np.linalg.solve(L, v)
    Te = (3*4/4) * Lm * (iqs*idr - ids*iqr)
    dwr_dt = (Te - B*wr) / J
    return np.array([*di_dt, dwr_dt])

def simulate_motor(params, t_span=[0, 2], n_points=500):
    """Simula el comportamiento del motor"""
    vqs, vds = 220*np.sqrt(2)/np.sqrt(3), 0
    try:
        sol = solve_ivp(lambda t, x: induction_motor(t, x, params, vqs, vds),
                        t_span, [0,0,0,0,0], dense_output=True, rtol=1e-6, atol=1e-8)
        t = np.linspace(t_span[0], t_span[1], n_points)
        iqs, ids, iqr, idr, wr = sol.sol(t)
        Is_mag = np.sqrt(iqs**2 + ids**2)
        Te = (3*4/4) * params[4] * (iqs*idr - ids*iqr)
        rpm = wr * 60/(2*np.pi) * 2/4
        power_factor = np.cos(np.arctan2(iqs, ids))
        return t, {'iqs': iqs, 'ids': ids, 'Is_mag': Is_mag, 'Te': Te,
                   'rpm': rpm, 'wr': wr, 'power_factor': power_factor}
    except Exception as e:
        print(f"Error en simulación: {e}")
        t = np.linspace(t_span[0], t_span[1], n_points)
        return t, {'iqs': np.ones(n_points)*1e6, 'ids': np.ones(n_points)*1e6,
                   'Is_mag': np.ones(n_points)*1e6, 'Te': np.ones(n_points)*1e6,
                   'rpm': np.ones(n_points)*1e6, 'wr': np.ones(n_points)*1e6,
                   'power_factor': np.ones(n_points)*1e6}

# ===============================================================================
# CARGA Y PROCESAMIENTO DE DATOS
# ===============================================================================

def load_and_process_data():
    """Carga y procesa los datos de los archivos CSV"""
    
    # Archivos disponibles
    files = {
        'PSO': 'PSO_adaptive_study.csv',
        'BFO': 'BFO_adaptive_study.csv', 
        'Chaotic PSO-DSO': 'Chaotic_PSODSO_adaptive_study.csv'
    }
    
    all_data = {}
    
    for algorithm, filename in files.items():
        if os.path.exists(filename):
            print(f"Cargando {filename}...")
            df = pd.read_csv(filename)
            
            # Procesar columnas con listas (identified_params, true_params)
            def safe_literal_eval(val):
                try:
                    if pd.isna(val):
                        return None
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return None
            
            df['identified_params'] = df['identified_params'].apply(safe_literal_eval)
            df['true_params'] = df['true_params'].apply(safe_literal_eval)
            
            # Filtrar filas válidas
            df = df.dropna(subset=['identified_params', 'true_params'])
            
            all_data[algorithm] = df
            print(f"   ✓ {len(df)} registros válidos cargados")
        else:
            print(f"   ⚠ Archivo {filename} no encontrado")
    
    return all_data

# ===============================================================================
# GRÁFICA 1: DESVIACIÓN ESTÁNDAR POR PARÁMETRO
# ===============================================================================

def plot_parameter_std_comparison(all_data):
    """Gráfica de desviación estándar por parámetro y algoritmo"""
    
    param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    error_columns = [f'error_{param}' for param in param_names]
    
    # Calcular estadísticas por algoritmo
    std_data = []
    mean_data = []
    
    for algorithm, df in all_data.items():
        # Agrupar por escenario y calcular estadísticas
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            
            for i, param in enumerate(param_names):
                error_col = error_columns[i]
                if error_col in scenario_data.columns:
                    errors = scenario_data[error_col].dropna()
                    if len(errors) > 0:
                        std_data.append({
                            'Algorithm': algorithm,
                            'Parameter': param,
                            'Scenario': scenario,
                            'Std_Error': np.std(errors),
                            'Mean_Error': np.mean(errors),
                            'Max_Error': np.max(errors),
                            'Min_Error': np.min(errors)
                        })
    
    std_df = pd.DataFrame(std_data)
    
    if len(std_df) == 0:
        print("No hay datos suficientes para la gráfica de desviación estándar")
        return
    
    # Crear gráfica
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Desviación Estándar por Parámetro del Motor', fontsize=16, fontweight='bold')
    
    # 1. Desviación estándar por parámetro (general)
    ax1 = axes[0, 0]
    std_summary = std_df.groupby(['Algorithm', 'Parameter'])['Std_Error'].mean().reset_index()
    pivot_std = std_summary.pivot(index='Parameter', columns='Algorithm', values='Std_Error')
    pivot_std.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Desviación Estándar Promedio por Parámetro')
    ax1.set_ylabel('Desviación Estándar (%)')
    ax1.legend(title='Algoritmo')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Error máximo por parámetro
    ax2 = axes[0, 1]
    max_summary = std_df.groupby(['Algorithm', 'Parameter'])['Max_Error'].mean().reset_index()
    pivot_max = max_summary.pivot(index='Parameter', columns='Algorithm', values='Max_Error')
    pivot_max.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Error Máximo Promedio por Parámetro')
    ax2.set_ylabel('Error Máximo (%)')
    ax2.legend(title='Algoritmo')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Comparación por escenario (desviación estándar)
    ax3 = axes[1, 0]
    scenario_std = std_df.groupby(['Algorithm', 'Scenario'])['Std_Error'].mean().reset_index()
    for algorithm in scenario_std['Algorithm'].unique():
        alg_data = scenario_std[scenario_std['Algorithm'] == algorithm]
        ax3.plot(alg_data['Scenario'], alg_data['Std_Error'], marker='o', label=algorithm, linewidth=2)
    ax3.set_title('Desviación Estándar por Escenario')
    ax3.set_ylabel('Desviación Estándar Promedio (%)')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Heatmap de errores por parámetro y algoritmo
    ax4 = axes[1, 1]
    heatmap_data = std_df.groupby(['Algorithm', 'Parameter'])['Mean_Error'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Parameter', columns='Algorithm', values='Mean_Error')
    im = ax4.imshow(heatmap_pivot.values, cmap='Reds', aspect='auto')
    ax4.set_xticks(range(len(heatmap_pivot.columns)))
    ax4.set_yticks(range(len(heatmap_pivot.index)))
    ax4.set_xticklabels(heatmap_pivot.columns)
    ax4.set_yticklabels(heatmap_pivot.index)
    ax4.set_title('Mapa de Calor: Error Medio por Parámetro')
    
    # Añadir valores en el heatmap
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            if not np.isnan(heatmap_pivot.iloc[i, j]):
                ax4.text(j, i, f'{heatmap_pivot.iloc[i, j]:.1f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=ax4, label='Error Medio (%)')
    
    plt.tight_layout()
    plt.savefig('analysis_plots/01_parameter_std_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica de desviación estándar guardada como '01_parameter_std_analysis.png'")

# ===============================================================================
# GRÁFICA 2: COMPARACIÓN DE RENDIMIENTO POR ALGORITMO
# ===============================================================================

def plot_algorithm_performance(all_data):
    """Comparación de rendimiento entre algoritmos"""
    
    # Combinar todos los datos
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if not combined_data:
        print("No hay datos para comparación de algoritmos")
        return
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación de Rendimiento entre Algoritmos', fontsize=16, fontweight='bold')
    
    # 1. Boxplot de errores por algoritmo
    ax1 = axes[0, 0]
    algorithms = combined_df['Algorithm'].unique()
    error_data = [combined_df[combined_df['Algorithm'] == alg]['error'].dropna() for alg in algorithms]
    bp1 = ax1.boxplot(error_data, labels=algorithms, patch_artist=True)
    colors = sns.color_palette("husl", len(algorithms))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title('Distribución de Errores por Algoritmo')
    ax1.set_ylabel('Error (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Tiempo de ejecución vs Error
    ax2 = axes[0, 1]
    for algorithm in algorithms:
        alg_data = combined_df[combined_df['Algorithm'] == algorithm]
        ax2.scatter(alg_data['time'], alg_data['error'], 
                   label=algorithm, alpha=0.7, s=60)
    ax2.set_xlabel('Tiempo de Ejecución (s)')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Eficiencia: Tiempo vs Error')
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Evaluaciones vs Error
    ax3 = axes[1, 0]
    for algorithm in algorithms:
        alg_data = combined_df[combined_df['Algorithm'] == algorithm]
        ax3.scatter(alg_data['evaluations'], alg_data['error'], 
                   label=algorithm, alpha=0.7, s=60)
    ax3.set_xlabel('Número de Evaluaciones')
    ax3.set_ylabel('Error (%)')
    ax3.set_title('Eficiencia: Evaluaciones vs Error')
    ax3.legend()
    ax3.set_yscale('log')
    
    # 4. Convergencia por algoritmo (si hay datos)
    ax4 = axes[1, 1]
    conv_data = combined_df[combined_df['convergence_evaluation'] > 0]
    if len(conv_data) > 0:
        conv_summary = conv_data.groupby('Algorithm')['convergence_evaluation'].agg(['mean', 'std', 'count'])
        algorithms_conv = conv_summary.index
        means = conv_summary['mean']
        stds = conv_summary['std']
        counts = conv_summary['count']
        
        x_pos = np.arange(len(algorithms_conv))
        ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(algorithms_conv, rotation=45)
        ax4.set_ylabel('Evaluaciones hasta Convergencia')
        ax4.set_title('Velocidad de Convergencia')
        
        # Añadir número de casos que convergieron
        for i, (mean, count) in enumerate(zip(means, counts)):
            ax4.text(i, mean + stds.iloc[i]/2, f'n={count}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No hay datos de convergencia disponibles', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Velocidad de Convergencia')
    
    plt.tight_layout()
    plt.savefig('analysis_plots/02_algorithm_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica de rendimiento de algoritmos guardada como '02_algorithm_performance.png'")

# ===============================================================================
# GRÁFICA 3: ANÁLISIS POR ESCENARIOS Y FASES
# ===============================================================================

def plot_scenario_phase_analysis(all_data):
    """Análisis detallado por escenarios y fases"""
    
    # Combinar datos
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if not combined_data:
        return
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis por Escenarios y Fases de Optimización', fontsize=16, fontweight='bold')
    
    # 1. Error promedio por escenario y fase
    ax1 = axes[0, 0]
    scenario_stats = combined_df.groupby(['Algorithm', 'scenario', 'phase'])['error'].mean().reset_index()
    
    scenarios = scenario_stats['scenario'].unique()
    x = np.arange(len(scenarios))
    width = 0.25
    
    algorithms = scenario_stats['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, algorithm in enumerate(algorithms):
        alg_data = scenario_stats[scenario_stats['Algorithm'] == algorithm]
        phase1_data = alg_data[alg_data['phase'] == 1].set_index('scenario')['error']
        phase2_data = alg_data[alg_data['phase'] == 2].set_index('scenario')['error']
        
        phase1_vals = [phase1_data.get(s, 0) for s in scenarios]
        phase2_vals = [phase2_data.get(s, 0) for s in scenarios]
        
        ax1.bar(x + i*width, phase1_vals, width, label=f'{algorithm} (Calibración)', 
               color=colors[i], alpha=0.7)
        ax1.bar(x + i*width, phase2_vals, width, bottom=phase1_vals, 
               label=f'{algorithm} (Adaptación)', color=colors[i], alpha=0.4)
    
    ax1.set_xlabel('Escenario')
    ax1.set_ylabel('Error Promedio (%)')
    ax1.set_title('Error por Escenario y Fase')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Evolución temporal por escenario
    ax2 = axes[0, 1]
    for algorithm in algorithms:
        for scenario in scenarios:
            scenario_data = combined_df[(combined_df['Algorithm'] == algorithm) & 
                                       (combined_df['scenario'] == scenario)]
            if len(scenario_data) > 0:
                scenario_data_sorted = scenario_data.sort_values('run')
                ax2.plot(scenario_data_sorted['run'], scenario_data_sorted['error'], 
                        marker='o', label=f'{algorithm}-{scenario}', alpha=0.7)
    
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Evolución del Error por Run')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_yscale('log')
    
    # 3. Distribución de errores por escenario
    ax3 = axes[1, 0]
    for i, scenario in enumerate(scenarios):
        scenario_errors = combined_df[combined_df['scenario'] == scenario]['error']
        ax3.hist(scenario_errors, bins=20, alpha=0.6, label=scenario, density=True)
    ax3.set_xlabel('Error (%)')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Distribución de Errores por Escenario')
    ax3.legend()
    ax3.set_xlim(0, combined_df['error'].quantile(0.95))
    
    # 4. Matriz de correlación de métricas
    ax4 = axes[1, 1]
    numeric_cols = ['cost', 'error', 'time', 'evaluations']
    available_cols = [col for col in numeric_cols if col in combined_df.columns]
    
    if len(available_cols) > 1:
        corr_matrix = combined_df[available_cols].corr()
        im = ax4.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(available_cols)))
        ax4.set_yticks(range(len(available_cols)))
        ax4.set_xticklabels(available_cols)
        ax4.set_yticklabels(available_cols)
        ax4.set_title('Matriz de Correlación de Métricas')
        
        # Añadir valores de correlación
        for i in range(len(available_cols)):
            for j in range(len(available_cols)):
                ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=10)
        
        plt.colorbar(im, ax=ax4, label='Correlación')
    else:
        ax4.text(0.5, 0.5, 'Datos insuficientes para correlación', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/03_scenario_phase_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica de análisis por escenarios guardada como '03_scenario_phase_analysis.png'")

# ===============================================================================
# GRÁFICAS: SIMULACIÓN DEL MOTOR POR ESCENARIOS (3 gráficas)
# ===============================================================================

def plot_motor_simulation_by_scenarios(all_data):
    """Simula y grafica el comportamiento del motor (torque, RPM, corriente) por escenario"""
    
    print("🔄 Iniciando simulación del motor por escenarios...")
    
    # Combinar datos
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if not combined_data:
        print("❌ No hay datos para simulación del motor")
        return
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    scenarios = combined_df['scenario'].unique()
    algorithms = combined_df['Algorithm'].unique()
    
    print(f"📊 Escenarios encontrados: {scenarios}")
    print(f"🤖 Algoritmos encontrados: {algorithms}")
    
    # Para cada escenario, generar una gráfica
    for scenario_idx, scenario in enumerate(scenarios):
        print(f"\n🔄 Procesando escenario: {scenario}")
        
        # Encontrar mejores parámetros por algoritmo para este escenario
        scenario_data = combined_df[combined_df['scenario'] == scenario]
        print(f"   📊 Datos en escenario: {len(scenario_data)} registros")
        
        best_cases = {}
        true_params = None
        
        for algorithm in algorithms:
            alg_data = scenario_data[scenario_data['Algorithm'] == algorithm]
            print(f"   🤖 {algorithm}: {len(alg_data)} registros")
            
            if len(alg_data) > 0:
                # Encontrar el mejor caso (menor error) para este algoritmo en este escenario
                best_idx = alg_data['error'].idxmin()
                best_case = alg_data.loc[best_idx]
                
                print(f"      📈 Mejor caso - Error: {best_case['error']:.2f}%")
                print(f"      🔧 Parámetros identificados: {best_case['identified_params']}")
                print(f"      ✅ Parámetros reales: {best_case['true_params']}")
                
                if best_case['identified_params'] is not None:
                    try:
                        # Verificar si ya es una lista o necesita conversión
                        if isinstance(best_case['identified_params'], str):
                            import ast
                            params_array = np.array(ast.literal_eval(best_case['identified_params']))
                        else:
                            params_array = np.array(best_case['identified_params'])
                            
                        best_cases[algorithm] = {
                            'params': params_array,
                            'error': best_case['error']
                        }
                        print(f"      ✅ Parámetros convertidos correctamente: {params_array}")
                    except Exception as e:
                        print(f"      ❌ Error convertiendo parámetros: {e}")
                        continue
                    
                    # Guardar parámetros verdaderos
                    if true_params is None and best_case['true_params'] is not None:
                        try:
                            if isinstance(best_case['true_params'], str):
                                true_params = np.array(ast.literal_eval(best_case['true_params']))
                            else:
                                true_params = np.array(best_case['true_params'])
                            print(f"      📋 Parámetros reales guardados: {true_params}")
                        except Exception as e:
                            print(f"      ⚠ Error procesando parámetros reales: {e}")
        
        if not best_cases:
            print(f"   ❌ No hay casos válidos para el escenario {scenario}")
            continue
            
        print(f"   ✅ Casos válidos encontrados: {list(best_cases.keys())}")
            
        # Simular motor para cada algoritmo
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'Comportamiento del Motor - Escenario: {scenario}', 
                     fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", len(best_cases))
        simulation_results = {}
        
        # Simular cada algoritmo
        for i, (algorithm, case) in enumerate(best_cases.items()):
            print(f"   🔬 Simulando {algorithm}...")
            try:
                t, outputs = simulate_motor(case['params'], t_span=[0, 2], n_points=1000)
                simulation_results[algorithm] = {
                    'time': t,
                    'outputs': outputs,
                    'error': case['error'],
                    'color': colors[i]
                }
                print(f"      ✅ Simulación exitosa - Error {case['error']:.2f}%")
            except Exception as e:
                print(f"      ❌ Error simulando {algorithm}: {e}")
                import traceback
                traceback.print_exc()
        
        # Simular con parámetros verdaderos
        true_simulation = None
        if true_params is not None:
            print("   🔬 Simulando con parámetros reales...")
            try:
                t_true, outputs_true = simulate_motor(true_params, t_span=[0, 2], n_points=1000)
                true_simulation = {'time': t_true, 'outputs': outputs_true}
                print("      ✅ Simulación con parámetros reales exitosa")
            except Exception as e:
                print(f"      ❌ Error en simulación real: {e}")
                import traceback
                traceback.print_exc()
        
        if not simulation_results:
            print(f"   ❌ No se pudieron realizar simulaciones para {scenario}")
            continue
        
        print("   📊 Creando gráficas...")
        
        # Gráfica 1: Corriente del Estator
        ax1 = axes[0]
        for algorithm, sim in simulation_results.items():
            ax1.plot(sim['time'], sim['outputs']['Is_mag'], 
                    label=f'{algorithm} (Error: {sim["error"]:.1f}%)', 
                    color=sim['color'], linewidth=2)
        
        if true_simulation:
            ax1.plot(true_simulation['time'], true_simulation['outputs']['Is_mag'], 
                    'k--', label='Parámetros Reales', linewidth=3, alpha=0.8)
        
        ax1.set_ylabel('Corriente del Estator (A)', fontsize=12, fontweight='bold')
        ax1.set_title('Magnitud de la Corriente del Estator', fontsize=13)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Gráfica 2: Torque Electromagnético
        ax2 = axes[1]
        for algorithm, sim in simulation_results.items():
            ax2.plot(sim['time'], sim['outputs']['Te'], 
                    label=f'{algorithm} (Error: {sim["error"]:.1f}%)', 
                    color=sim['color'], linewidth=2)
        
        if true_simulation:
            ax2.plot(true_simulation['time'], true_simulation['outputs']['Te'], 
                    'k--', label='Parámetros Reales', linewidth=3, alpha=0.8)
        
        ax2.set_ylabel('Torque Electromagnético (N·m)', fontsize=12, fontweight='bold')
        ax2.set_title('Torque Electromagnético', fontsize=13)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Gráfica 3: Velocidad (RPM)
        ax3 = axes[2]
        for algorithm, sim in simulation_results.items():
            ax3.plot(sim['time'], sim['outputs']['rpm'], 
                    label=f'{algorithm} (Error: {sim["error"]:.1f}%)', 
                    color=sim['color'], linewidth=2)
        
        if true_simulation:
            ax3.plot(true_simulation['time'], true_simulation['outputs']['rpm'], 
                    'k--', label='Parámetros Reales', linewidth=3, alpha=0.8)
        
        ax3.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Velocidad (RPM)', fontsize=12, fontweight='bold')
        ax3.set_title('Velocidad del Motor', fontsize=13)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfica por escenario
        filename = f'analysis_plots/motor_simulation_{scenario_idx+1:02d}_{scenario.replace(" ", "_").replace("_", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Gráfica guardada: {filename}")
        
        # Mostrar estadísticas del escenario
        print(f"   📊 Resumen del escenario {scenario}:")
        if true_params is not None:
            param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
            print(f"      📋 Parámetros reales: {dict(zip(param_names, true_params))}")
        
        for algorithm, case in best_cases.items():
            print(f"      🤖 {algorithm}: Error {case['error']:.2f}%")
        print()
    
    # Resumen comparativo final
    print(f"{'='*80}")
    print("📊 RESUMEN COMPARATIVO POR ESCENARIOS")
    print(f"{'='*80}")
    print(f"{'Escenario':<25} | {'Algoritmo':<20} | {'Error (%)':<10}")
    print("-"*80)
    
    for scenario in scenarios:
        scenario_data = combined_df[combined_df['scenario'] == scenario]
        print(f"{scenario:<25} |")
        
        for algorithm in algorithms:
            alg_data = scenario_data[scenario_data['Algorithm'] == algorithm]
            if len(alg_data) > 0:
                best_error = alg_data['error'].min()
                print(f"{'':>26} | {algorithm:<20} | {best_error:<10.2f}")
        print("-"*80)

# ===============================================================================
# GRÁFICA 5: ANÁLISIS ESTADÍSTICO AVANZADO
# ===============================================================================

def plot_advanced_statistical_analysis(all_data):
    """Análisis estadístico avanzado de los resultados"""
    
    # Combinar datos
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if not combined_data:
        return
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Estadístico Avanzado de Resultados', fontsize=16, fontweight='bold')
    
    # 1. Test de normalidad (Q-Q plots)
    ax1 = axes[0, 0]
    algorithms = combined_df['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, algorithm in enumerate(algorithms):
        alg_errors = combined_df[combined_df['Algorithm'] == algorithm]['error'].dropna()
        if len(alg_errors) > 3:
            # Usar scatter plot en lugar de probplot con argumentos no soportados
            from scipy import stats as scipy_stats
            theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(alg_errors)))
            sample_quantiles = np.sort(alg_errors)
            ax1.scatter(theoretical_quantiles, sample_quantiles, 
                       color=colors[i], alpha=0.7, label=algorithm)
    
    ax1.set_title('Q-Q Plot: Normalidad de Errores')
    ax1.set_xlabel('Cuantiles Teóricos')
    ax1.set_ylabel('Cuantiles de Muestra')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Intervalos de confianza
    ax2 = axes[0, 1]
    conf_data = []
    for algorithm in algorithms:
        alg_data = combined_df[combined_df['Algorithm'] == algorithm]['error'].dropna()
        if len(alg_data) > 1:
            mean_err = np.mean(alg_data)
            ci = stats.t.interval(0.95, len(alg_data)-1, 
                                 loc=mean_err, 
                                 scale=stats.sem(alg_data))
            conf_data.append({
                'Algorithm': algorithm,
                'Mean': mean_err,
                'CI_Lower': ci[0],
                'CI_Upper': ci[1],
                'Std': np.std(alg_data)
            })
    
    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        x_pos = np.arange(len(conf_df))
        
        ax2.bar(x_pos, conf_df['Mean'], yerr=[conf_df['Mean'] - conf_df['CI_Lower'], 
                                              conf_df['CI_Upper'] - conf_df['Mean']], 
               capsize=5, alpha=0.7, color=colors[:len(conf_df)])
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(conf_df['Algorithm'], rotation=45)
        ax2.set_ylabel('Error Medio (%)')
        ax2.set_title('Intervalos de Confianza (95%)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Análisis de Pareto (80/20)
    ax3 = axes[0, 2]
    # Ordenar errores y calcular porcentaje acumulativo
    all_errors = combined_df['error'].dropna().sort_values(ascending=False)
    cumsum_pct = (np.cumsum(all_errors) / np.sum(all_errors)) * 100
    case_pct = np.arange(1, len(all_errors) + 1) / len(all_errors) * 100
    
    ax3.plot(case_pct, cumsum_pct, 'b-', linewidth=2, label='Error Acumulativo')
    ax3.axhline(y=80, color='r', linestyle='--', label='80%')
    ax3.axvline(x=20, color='r', linestyle='--', label='20% de casos')
    ax3.set_xlabel('Porcentaje de Casos (%)')
    ax3.set_ylabel('Error Acumulativo (%)')
    ax3.set_title('Análisis de Pareto')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución de residuos
    ax4 = axes[1, 0]
    for i, algorithm in enumerate(algorithms):
        alg_errors = combined_df[combined_df['Algorithm'] == algorithm]['error'].dropna()
        if len(alg_errors) > 0:
            # Calcular residuos (diferencia con la mediana)
            residuals = alg_errors - np.median(alg_errors)
            ax4.scatter(np.random.normal(i, 0.1, len(residuals)), residuals, 
                       alpha=0.6, color=colors[i], label=algorithm)
    
    ax4.set_xticks(range(len(algorithms)))
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.set_ylabel('Residuos (%)')
    ax4.set_title('Distribución de Residuos')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. Test de significancia estadística
    ax5 = axes[1, 1]
    if len(algorithms) >= 2:
        # ANOVA test
        groups = [combined_df[combined_df['Algorithm'] == alg]['error'].dropna().values 
                 for alg in algorithms]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Crear gráfica de significancia
                significance_data = []
                for i in range(len(algorithms)):
                    for j in range(i+1, len(algorithms)):
                        if len(groups[i]) > 0 and len(groups[j]) > 0:
                            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                            significance_data.append({
                                'Comparison': f'{algorithms[i]} vs {algorithms[j]}',
                                'p_value': p_val,
                                'significant': p_val < 0.05
                            })
                
                if significance_data:
                    sig_df = pd.DataFrame(significance_data)
                    y_pos = np.arange(len(sig_df))
                    colors_sig = ['red' if sig else 'green' for sig in sig_df['significant']]
                    
                    ax5.barh(y_pos, -np.log10(sig_df['p_value']), color=colors_sig, alpha=0.7)
                    ax5.set_yticks(y_pos)
                    ax5.set_yticklabels(sig_df['Comparison'])
                    ax5.set_xlabel('-log10(p-value)')
                    ax5.set_title('Significancia Estadística\n(Rojo: p<0.05)')
                    ax5.axvline(x=-np.log10(0.05), color='black', linestyle='--', 
                               label='p=0.05')
                    ax5.legend()
                    
                    # Añadir texto con ANOVA
                    ax5.text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}', 
                            transform=ax5.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax5.text(0.5, 0.5, 'Datos insuficientes\npara análisis de significancia', 
                            ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('Significancia Estadística')
                    
            except Exception as e:
                ax5.text(0.5, 0.5, f'Error en análisis estadístico:\n{e}', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Significancia Estadística')
    else:
        ax5.text(0.5, 0.5, 'Se requieren al menos 2 algoritmos\npara comparación estadística', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Significancia Estadística')
    
    # 6. Métricas de robustez
    ax6 = axes[1, 2]
    robustness_metrics = []
    for algorithm in algorithms:
        alg_errors = combined_df[combined_df['Algorithm'] == algorithm]['error'].dropna()
        if len(alg_errors) > 1:
            # Coeficiente de variación
            cv = np.std(alg_errors) / np.mean(alg_errors)
            # Rango intercuartílico relativo
            q75, q25 = np.percentile(alg_errors, [75, 25])
            iqr_relative = (q75 - q25) / np.median(alg_errors)
            robustness_metrics.append({
                'Algorithm': algorithm,
                'CV': cv,
                'IQR_Relative': iqr_relative
            })
    
    if robustness_metrics:
        rob_df = pd.DataFrame(robustness_metrics)
        x_pos = np.arange(len(rob_df))
        width = 0.35
        
        ax6.bar(x_pos - width/2, rob_df['CV'], width, label='Coef. Variación', alpha=0.7)
        ax6.bar(x_pos + width/2, rob_df['IQR_Relative'], width, label='IQR Relativo', alpha=0.7)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(rob_df['Algorithm'], rotation=45)
        ax6.set_ylabel('Métrica de Robustez')
        ax6.set_title('Métricas de Robustez\n(Menor = Más Robusto)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/05_advanced_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Análisis estadístico avanzado guardado como '05_advanced_statistical_analysis.png'")

# ===============================================================================
# GRÁFICAS 6-12: ANÁLISIS INDIVIDUAL POR PARÁMETRO (7 gráficas simples)
# ===============================================================================

def plot_individual_parameter_analysis(all_data):
    """Genera 7 gráficas individuales, una por cada parámetro del motor
    Eje X: Algoritmos, Eje Y: Valores identificados, Línea punteada: Valor real"""
    
    param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    param_labels = ['Resistencia Estator (rs)', 'Resistencia Rotor (rr)', 
                   'Inductancia Dispersión Estator (Lls)', 'Inductancia Dispersión Rotor (Llr)',
                   'Inductancia Magnetización (Lm)', 'Inercia (J)', 'Fricción (B)']
    param_units = ['Ω', 'Ω', 'H', 'H', 'H', 'kg·m²', 'N·m·s']
    
    # Combinar datos
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if not combined_data:
        print("No hay datos para análisis individual de parámetros")
        return
        
    combined_df = pd.concat(combined_data, ignore_index=True)
    algorithms = combined_df['Algorithm'].unique()
    
    print(f"\n📊 Generando {len(param_names)} gráficas individuales por parámetro...")
    
    # Generar una gráfica por cada parámetro
    for idx, (param, label, unit) in enumerate(zip(param_names, param_labels, param_units)):
        
        # Extraer datos del parámetro específico
        param_data = []
        true_value = None
        
        for _, row in combined_df.iterrows():
            if (row['identified_params'] is not None and 
                row['true_params'] is not None):
                
                identified = np.array(row['identified_params'])
                true_vals = np.array(row['true_params'])
                
                if idx < len(identified) and idx < len(true_vals):
                    param_data.append({
                        'Algorithm': row['Algorithm'],
                        'Scenario': row['scenario'],
                        'Run': row['run'],
                        'Identified_Value': identified[idx],
                        'True_Value': true_vals[idx]
                    })
                    
                    # Guardar el valor verdadero (debería ser el mismo en todos)
                    if true_value is None:
                        true_value = true_vals[idx]
        
        if not param_data:
            print(f"   ⚠ No hay datos válidos para el parámetro {param}")
            continue
            
        param_df = pd.DataFrame(param_data)
        
        # Crear la gráfica
        plt.figure(figsize=(12, 8))
        
        # Preparar datos por algoritmo
        colors = sns.color_palette("husl", len(algorithms))
        x_positions = np.arange(len(algorithms))
        
        # Para cada algoritmo, obtener todos los valores identificados
        for i, algorithm in enumerate(algorithms):
            alg_data = param_df[param_df['Algorithm'] == algorithm]['Identified_Value']
            
            if len(alg_data) > 0:
                # Agregar un poco de jitter en X para separar puntos
                x_jittered = np.random.normal(i, 0.1, len(alg_data))
                
                # Scatter plot de todos los runs
                plt.scatter(x_jittered, alg_data, 
                           alpha=0.7, s=80, color=colors[i], 
                           label=f'{algorithm} (n={len(alg_data)})')
                
                # Agregar media y desviación estándar
                mean_val = np.mean(alg_data)
                std_val = np.std(alg_data)
                
                plt.errorbar(i, mean_val, yerr=std_val, 
                           fmt='D', color='black', markersize=8, 
                           capsize=5, capthick=2, elinewidth=2)
                
                # Añadir texto con valor medio
                plt.text(i, mean_val + std_val + (true_value * 0.05), 
                        f'{mean_val:.4f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Línea horizontal punteada con el valor real
        if true_value is not None:
            plt.axhline(y=true_value, color='red', linestyle='--', linewidth=3, 
                       label=f'Valor Real: {true_value:.4f} {unit}', alpha=0.8)
        
        # Configurar ejes y etiquetas
        plt.xticks(x_positions, algorithms)
        plt.xlabel('Algoritmo de Optimización', fontsize=12, fontweight='bold')
        plt.ylabel(f'{label} ({unit})', fontsize=12, fontweight='bold')
        plt.title(f'Identificación del Parámetro: {param} ({label})', 
                 fontsize=14, fontweight='bold')
        
        # Leyenda
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Grid y estilo
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Calcular y mostrar estadísticas de error
        if true_value is not None:
            errors_by_algorithm = []
            for algorithm in algorithms:
                alg_data = param_df[param_df['Algorithm'] == algorithm]['Identified_Value']
                if len(alg_data) > 0:
                    mean_error = np.mean(np.abs(alg_data - true_value) / true_value * 100)
                    errors_by_algorithm.append((algorithm, mean_error))
            
            # Ordenar por menor error
            errors_by_algorithm.sort(key=lambda x: x[1])
            
            # Añadir texto con ranking de errores
            error_text = "Ranking por Error Medio:\n"
            for i, (alg, error) in enumerate(errors_by_algorithm):
                error_text += f"{i+1}. {alg}: {error:.2f}%\n"
            
            plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        # Guardar gráfica
        filename = f'analysis_plots/param_{idx+6:02d}_{param}_identification.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Gráfica del parámetro {param} guardada como '{filename}'")
        
        # Mostrar estadísticas en consola
        if true_value is not None:
            print(f"   📈 Parámetro {param} - Valor real: {true_value:.4f} {unit}")
            for algorithm in algorithms:
                alg_data = param_df[param_df['Algorithm'] == algorithm]['Identified_Value']
                if len(alg_data) > 0:
                    mean_val = np.mean(alg_data)
                    mean_error = np.mean(np.abs(alg_data - true_value) / true_value * 100)
                    print(f"      {algorithm:>20}: {mean_val:.4f} {unit} (Error: {mean_error:.2f}%)")
            print()
    
    print(f"{'='*80}")
    print("RESUMEN DE IDENTIFICACIÓN POR PARÁMETROS")
    print(f"{'='*80}")

# ===============================================================================
# FUNCIÓN PRINCIPAL
# ===============================================================================

def main():
    """Función principal que ejecuta todo el análisis"""
    
    print("="*80)
    print("ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS - DIGITAL TWIN ADAPTATIVO")
    print("="*80)
    
    # Cargar datos
    all_data = load_and_process_data()
    
    if not all_data:
        print("❌ No se encontraron datos válidos para analizar")
        return
    
    print(f"\n✓ Datos cargados exitosamente para {len(all_data)} algoritmos")
    for algorithm, df in all_data.items():
        print(f"   - {algorithm}: {len(df)} registros")
    
    print(f"\n📊 Generando gráficas de análisis...")
    
    # Generar todas las gráficas
    try:
        plot_parameter_std_comparison(all_data)
    except Exception as e:
        print(f"❌ Error en gráfica 1: {e}")
    
    try:
        plot_algorithm_performance(all_data)
    except Exception as e:
        print(f"❌ Error en gráfica 2: {e}")
    
    try:
        plot_scenario_phase_analysis(all_data)
    except Exception as e:
        print(f"❌ Error en gráfica 3: {e}")
    
    try:
        plot_motor_simulation_best_case(all_data)
    except Exception as e:
        print(f"❌ Error en gráfica 4: {e}")
    
    try:
        plot_advanced_statistical_analysis(all_data)
    except Exception as e:
        print(f"❌ Error en gráfica 5: {e}")
    
    # Generar las 7 gráficas individuales por parámetro
    try:
        plot_individual_parameter_analysis(all_data)
    except Exception as e:
        print(f"❌ Error en gráficas individuales de parámetros: {e}")
    
    print("\n" + "="*80)
    print("🎯 ANÁLISIS COMPLETADO")
    print("="*80)
    print("\n📁 Gráficas generadas en la carpeta 'analysis_plots/':")
    print("   • 01_parameter_std_analysis.png - Análisis de desviación estándar")
    print("   • 02_algorithm_performance.png - Comparación de rendimiento")
    print("   • 03_scenario_phase_analysis.png - Análisis por escenarios")
    print("   • 04_motor_simulation_best_case.png - Simulación del motor")
    print("   • 05_advanced_statistical_analysis.png - Análisis estadístico avanzado")
    print("\n📊 Análisis individual por parámetro (7 gráficas):")
    print("   • param_06_rs_detailed_analysis.png - Resistencia del Estator")
    print("   • param_07_rr_detailed_analysis.png - Resistencia del Rotor")
    print("   • param_08_Lls_detailed_analysis.png - Inductancia Dispersión Estator")
    print("   • param_09_Llr_detailed_analysis.png - Inductancia Dispersión Rotor")
    print("   • param_10_Lm_detailed_analysis.png - Inductancia de Magnetización")
    print("   • param_11_J_detailed_analysis.png - Inercia")
    print("   • param_12_B_detailed_analysis.png - Fricción")
    
    # Estadísticas finales
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    combined_data = []
    for algorithm, df in all_data.items():
        df_copy = df.copy()
        df_copy['Algorithm'] = algorithm
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"   • Total de experimentos analizados: {len(combined_df)}")
        print(f"   • Error promedio general: {combined_df['error'].mean():.2f}%")
        print(f"   • Error mínimo encontrado: {combined_df['error'].min():.2f}%")
        print(f"   • Tiempo promedio de ejecución: {combined_df['time'].mean():.1f}s")
        print(f"   • Evaluaciones promedio: {combined_df['evaluations'].mean():.0f}")

if __name__ == "__main__":
    main()