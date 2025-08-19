#!/usr/bin/env python3
"""
Análisis Gemelo Digital Adaptativo - Sistema de 2 Fases
FASE 1: Normal Operation (Multi-señal) - Calibración
FASE 2: High Temperature + Severe Conditions (Solo corriente) - Adaptación

Autor: Tu nombre
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

class AdaptiveAnalyzer:
    def __init__(self):
        self.algorithms = ['BFO', 'PSO', 'Chaotic_PSODSO']
        self.param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
        self.data = self.load_data()
        
    def load_data(self):
        """Cargar datos CSV"""
        print("📂 Cargando datos...")
        
        combined_data = []
        file_mapping = {
            'BFO': 'BFO_adaptive_results.csv',
            'PSO': 'PSO_adaptive_results.csv',
            'Chaotic_PSODSO': 'Chaotic_PSODSO_adaptive_results.csv'
        }
        
        for alg_name, filename in file_mapping.items():
            try:
                df = pd.read_csv(filename)
                df['algorithm'] = alg_name
                combined_data.append(df)
                print(f"  ✓ {alg_name}: {len(df)} filas")
            except FileNotFoundError:
                print(f"  ✗ {filename} no encontrado")
        
        if not combined_data:
            raise FileNotFoundError("No se encontraron archivos CSV. Asegúrate de que los archivos estén en el directorio correcto.")

        data = pd.concat(combined_data, ignore_index=True)
        print(f"📊 Total: {len(data)} observaciones\n")
        return data
    
    def analyze_adaptability(self):
        """ANÁLISIS PRINCIPAL: Adaptabilidad Fase 1 vs Fase 2"""
        print("🔥 ANÁLISIS DE ADAPTABILIDAD")
        print("="*50)
        
        results = {}
        
        for algorithm in self.algorithms:
            print(f"\n🔍 {algorithm}:")
            
            phase1 = self.data[(self.data['algorithm'] == algorithm) & (self.data['scenario'] == 'Normal_Operation')]
            phase2 = self.data[(self.data['algorithm'] == algorithm) & (self.data['scenario'].isin(['High_Temperature', 'Severe_Conditions']))]
            
            if len(phase1) == 0 or len(phase2) == 0:
                print("  ⚠️ Datos insuficientes")
                continue
            
            phase1_error = phase1['error'].mean()
            phase2_error = phase2['error'].mean()
            error_degradation = ((phase2_error - phase1_error) / phase1_error) * 100 if phase1_error != 0 else float('inf')
            adaptability_score = max(0, 100 - abs(error_degradation))
            
            results[algorithm] = {'adaptability_score': adaptability_score, 'error_degradation': error_degradation}
            
            print(f"  📊 Error: {phase1_error:.1f}% → {phase2_error:.1f}% ({error_degradation:+.1f}%)")
            print(f"  🏆 Score adaptabilidad: {adaptability_score:.1f}")
        
        ranking = sorted(results.items(), key=lambda x: x[1]['adaptability_score'], reverse=True)
        
        print(f"\n🏆 RANKING DE ADAPTABILIDAD:")
        for i, (alg, data) in enumerate(ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {medal} {alg}: {data['adaptability_score']:.1f} (degradación: {data['error_degradation']:+.1f}%)")
        
        return results
    
    def analyze_parameters(self):
        """ANÁLISIS DE PARÁMETROS DQ: Cuáles son más problemáticos"""
        print(f"\n📊 ANÁLISIS DE PARÁMETROS DQ")
        print("="*50)
        
        param_difficulty = {}
        for param in self.param_names:
            error_col = f'error_{param}'
            if error_col in self.data.columns:
                avg_error = self.data[error_col].mean()
                param_difficulty[param] = {'avg_error': avg_error}
        
        ranking = sorted(param_difficulty.items(), key=lambda x: x[1]['avg_error'], reverse=True)
        
        print(f"\n🎯 PARÁMETROS MÁS PROBLEMÁTICOS:")
        for i, (param, data) in enumerate(ranking):
            difficulty = 'Alto' if data['avg_error'] > 20 else 'Medio' if data['avg_error'] > 10 else 'Bajo'
            icon = '🔴' if difficulty == 'Alto' else '🟡' if difficulty == 'Medio' else '🟢'
            print(f"  {icon} {i+1}. {param}: {data['avg_error']:.1f}% promedio ({difficulty})")
        
        return param_difficulty
    
    def compare_algorithms(self):
        """COMPARACIÓN GENERAL DE ALGORITMOS"""
        print(f"\n⚡ COMPARACIÓN GENERAL DE ALGORITMOS")
        print("="*50)
        
        comparison = {}
        for algorithm in self.algorithms:
            alg_data = self.data[self.data['algorithm'] == algorithm]
            if alg_data.empty: continue
            mean_error = alg_data['error'].mean()
            mean_time = alg_data['time'].mean()
            std_error = alg_data['error'].std()
            robustness = 1 / (std_error / mean_error + 1e-6) if mean_error > 0 else 0
            accuracy_score = max(0, 100 - mean_error)
            speed_score = max(0, 100 - mean_time / 10)
            overall_score = (accuracy_score * 0.5 + speed_score * 0.3 + robustness * 2)
            comparison[algorithm] = {'overall_score': overall_score}
            print(f"\n🔍 {algorithm}: Score general: {overall_score:.1f}")
        return comparison

    def statistical_analysis(self):
        """ACTUALIZADO: Análisis estadístico con ANOVA y test post-hoc de Tukey."""
        print(f"\n📈 ANÁLISIS ESTADÍSTICO AVANZADO")
        print("="*50)
        
        # 1. ANOVA para diferencias generales entre algoritmos
        algorithm_groups = [self.data[self.data['algorithm'] == alg]['error'].dropna() for alg in self.algorithms]
        algorithm_groups_clean = [g for g in algorithm_groups if len(g) > 1]

        if len(algorithm_groups_clean) < 2:
            print("\n🔬 ANOVA y Tukey - No hay suficientes datos para comparar algoritmos.")
            return

        f_stat, p_value = f_oneway(*algorithm_groups_clean)
        print(f"\n🔬 ANOVA - Diferencias generales entre algoritmos:")
        print(f"  P-value: {p_value:.6f}")
        
        # 2. Test Post-Hoc de Tukey si ANOVA es significativo
        if p_value < 0.05:
            print("  ✓ Diferencias significativas encontradas. Realizando test de Tukey HSD...")
            
            # Preparar los datos para el test de Tukey
            all_data = pd.concat(algorithm_groups_clean)
            labels = [alg for alg, g in zip(self.algorithms, algorithm_groups) for _ in range(len(g))]
            
            tukey_results = pairwise_tukeyhsd(endog=all_data, groups=labels, alpha=0.05)
            
            print("\n🔍 Resultados del Test de Tukey (comparaciones por pares):")
            print(tukey_results)
            print("\n  (La columna 'reject=True' indica que la diferencia entre esos dos algoritmos es estadísticamente significativa)")

        else:
            print("  ✗ Sin diferencias estadísticas significativas entre los algoritmos.")

    def create_visualizations(self):
        """CREAR VISUALIZACIONES CLAVE (DASHBOARD)"""
        print(f"\n📊 CREANDO DASHBOARD DE ANÁLISIS...")
        plt.style.use('default')
        sns.set_palette("husl")
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle('Análisis de Gemelo Digital Adaptativo', fontsize=18, fontweight='bold')
        sns.barplot(data=self.data, x='algorithm', y='error', hue='scenario', ax=axes[0,0], palette="viridis")
        axes[0,0].set_title('Error Promedio por Fase y Algoritmo')
        sns.barplot(data=self.data, x='algorithm', y='time', ax=axes[0,1], palette="flare")
        axes[0,1].set_title('Tiempo de Optimización Promedio')
        error_param_cols = [f'error_{param}' for param in self.param_names]
        existing_error_cols = [col for col in error_param_cols if col in self.data.columns]
        if existing_error_cols:
            param_matrix = self.data.groupby('algorithm')[existing_error_cols].mean()
            param_matrix.columns = [col.replace('error_', '') for col in param_matrix.columns]
            sns.heatmap(param_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1,0])
        axes[1,0].set_title('Error Promedio por Parámetro DQ (%)')
        success_df = self.data.groupby('algorithm').apply(lambda x: (x['error'] < 5).mean() * 100).reset_index(name='Success_Rate').sort_values('Success_Rate')
        bars = axes[1,1].barh(success_df['algorithm'], success_df['Success_Rate'], color=sns.color_palette("rocket", len(success_df)))
        axes[1,1].set_title('Tasa de Éxito (<5% error)')
        axes[1,1].set_xlim(0, 105)
        for bar in bars: axes[1,1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}%', va='center')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('adaptive_analysis_dashboard.png', dpi=300)
        plt.show()

    def create_scenario_comparison_plot(self):
        """ACTUALIZADO: Crea un Gráfico de Violín para comparar el rendimiento en cada escenario."""
        print(f"\n🎻 CREANDO GRÁFICO DE VIOLÍN POR ESCENARIO...")
        scenarios = ['Normal_Operation', 'High_Temperature', 'Severe_Conditions']
        plot_data = self.data[self.data['scenario'].isin(scenarios)]
        if plot_data.empty: return
        
        plt.figure(figsize=(14, 8))
        
        sns.violinplot(data=plot_data, x='algorithm', y='error', hue='scenario', 
                       palette='viridis', inner='quartile', split=True, cut=0)
        
        plt.title('Distribución del Error del Algoritmo por Escenario', fontsize=16, fontweight='bold')
        plt.ylabel('Distribución del Error de Estimación (%)')
        plt.xlabel('Algoritmo')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Condición de Operación')
        plt.tight_layout()
        plt.savefig('scenario_violin_comparison.png', dpi=300)
        plt.show()
        print("  ✓ Gráfico de violín guardado: scenario_violin_comparison.png")

    def create_spider_dashboard(self):
        """Crea un dashboard con gráficos de araña de 6 métricas para cada escenario."""
        print(f"\n🕸️ CREANDO DASHBOARD DE GRÁFICOS DE ARAÑA POR ESCENARIO...")
        
        scenarios = ['Normal_Operation', 'High_Temperature', 'Severe_Conditions']
        metrics = ['Precisión (Prom.)', 'Precisión (Típica)', 'Consistencia', 'Estabilidad', 'Velocidad', 'Eficiencia']
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw=dict(polar=True))
        fig.suptitle('Perfil de Rendimiento del Algoritmo por Escenario', fontsize=20, fontweight='bold')

        for ax, scenario in zip(axes, scenarios):
            scenario_data = self.data[self.data['scenario'] == scenario]
            if scenario_data.empty:
                ax.text(0.5, 0.5, 'Sin datos', horizontalalignment='center', verticalalignment='center')
                ax.set_title(scenario.replace('_', ' '), fontsize=14, y=1.1)
                continue

            stats_by_alg = {}
            for alg in self.algorithms:
                alg_data = scenario_data[scenario_data['algorithm'] == alg]
                if alg_data.empty: continue
                
                mean_error = alg_data['error'].mean()
                median_error = alg_data['error'].median()
                std_error = alg_data['error'].std()
                max_error = alg_data['error'].max()
                mean_time = alg_data['time'].mean()
                
                precision_avg = max(0, 100 - mean_error)
                precision_median = max(0, 100 - median_error)
                consistency = 100 / (1 + (std_error / mean_error)) if mean_error > 0 else 0
                stability = max(0, 100 - max_error)
                speed = 1000 / mean_time if mean_time > 0 else 0
                efficiency = precision_avg / mean_time if mean_time > 0 else 0
                
                stats_by_alg[alg] = [precision_avg, precision_median, consistency, stability, speed, efficiency]

            df_stats = pd.DataFrame.from_dict(stats_by_alg, orient='index', columns=metrics)
            if df_stats.empty: continue
            
            df_max = df_stats.max()
            df_normalized = df_stats.copy()
            for col in df_normalized.columns:
                if df_max[col] > 0:
                    df_normalized[col] = (df_normalized[col] / df_max[col]) * 100

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            for alg in df_normalized.index:
                values = df_normalized.loc[alg].tolist()
                values += values[:1]
                ax.plot(angles, values, label=alg, linewidth=2)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
            ax.set_title(scenario.replace('_', ' '), fontsize=14, y=1.1)
            ax.set_rlabel_position(180)

        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('spider_dashboard.png', dpi=300)
        plt.show()
        print("  ✓ Dashboard de araña guardado: spider_dashboard.png")
        
    def run_complete_analysis(self):
        """EJECUTAR ANÁLISIS COMPLETO"""
        print("🚀 SISTEMA DE ANÁLISIS DE GEMELO DIGITAL ADAPTATIVO")
        print("="*60)
        
        self.analyze_adaptability()
        self.analyze_parameters()
        self.compare_algorithms()
        self.statistical_analysis()
        self.create_visualizations()
        self.create_scenario_comparison_plot()
        self.create_spider_dashboard()
        
        print(f"\n✅ ANÁLISIS COMPLETADO")

# ===== EJECUCIÓN PRINCIPAL =====
if __name__ == "__main__":
    print("🔬 Iniciando Análisis de Gemelo Digital Adaptativo...")
    
    try:
        analyzer = AdaptiveAnalyzer()
        analyzer.run_complete_analysis()
        
        print(f"\n🎉 ¡Análisis completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que los archivos CSV ('BFO_adaptive_results.csv', etc.) estén en el mismo directorio.")
