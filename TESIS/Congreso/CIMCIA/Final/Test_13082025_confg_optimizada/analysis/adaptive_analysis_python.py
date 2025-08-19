#!/usr/bin/env python3
"""
Sistema Modular de Análisis: Gemelo Digital Adaptativo
Análisis de algoritmos bio-inspirados para identificación de parámetros DQ

Módulos:
1. Análisis de Adaptabilidad (Fase 1 vs Fase 2)
2. Heatmap de Parámetros DQ
3. Dashboard Comparativo
4. Análisis Estadístico Robusto
5. Análisis de Convergencia

Autor: [Tu Nombre]
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, shapiro, levene
import warnings
import os
from pathlib import Path
import json

warnings.filterwarnings('ignore')

class AdaptiveDigitalTwinAnalyzer:
    """Sistema modular para análisis de gemelo digital adaptativo"""
    
    def __init__(self, data_path="./"):
        """
        Inicializar analizador
        
        Args:
            data_path (str): Ruta donde están los archivos CSV
        """
        self.data_path = Path(data_path)
        self.algorithms = ['BFO', 'PSO', 'Chaotic_PSODSO']
        self.param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
        self.scenarios = ['Normal_Operation', 'High_Temperature', 'Severe_Conditions']
        self.phases = [1, 2]
        
        # Cargar datos
        self.data = self._load_data()
        
        # Crear directorio de resultados
        self.results_path = Path("results_analysis")
        self.results_path.mkdir(exist_ok=True)
        (self.results_path / "plots").mkdir(exist_ok=True)
        (self.results_path / "tables").mkdir(exist_ok=True)
        
        print("🔬 Sistema de Análisis Modular Inicializado")
        print(f"📊 Datos cargados: {len(self.data)} observaciones")
        print(f"🧬 Algoritmos: {', '.join(self.algorithms)}")
        print(f"⚙️ Parámetros DQ: {', '.join(self.param_names)}")
        
    def _load_data(self):
        """Cargar y combinar datos CSV de todos los algoritmos"""
        combined_data = []
        
        file_mapping = {
            'BFO': 'BFO_adaptive_results.csv',
            'PSO': 'PSO_adaptive_results.csv',
            'Chaotic_PSODSO': 'Chaotic_PSODSO_adaptive_results.csv'
        }
        
        for alg_name, filename in file_mapping.items():
            file_path = self.data_path / filename
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['algorithm'] = alg_name
                combined_data.append(df)
                print(f"✓ Cargado {alg_name}: {len(df)} filas")
            else:
                print(f"⚠️ Archivo no encontrado: {filename}")
        
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            raise FileNotFoundError("No se encontraron archivos CSV válidos")
    
    # ===== MÓDULO 1: ANÁLISIS DE ADAPTABILIDAD =====
    def module_1_adaptability_analysis(self):
        """
        MÓDULO 1: Análisis de Adaptabilidad
        Compara rendimiento entre Fase 1 (Calibración) y Fase 2 (Adaptación)
        """
        print("\n" + "="*80)
        print("🔥 MÓDULO 1: ANÁLISIS DE ADAPTABILIDAD")
        print("Objetivo: Comparar rendimiento Fase 1 vs Fase 2")
        print("="*80)
        
        results = {
            'title': 'Análisis de Adaptabilidad',
            'description': 'Evaluación de degradación al pasar de multi-señal a solo corriente',
            'phase_comparison': {},
            'degradation_metrics': {},
            'parameter_impact': {},
            'adaptability_ranking': [],
            'insights': [],
            'summary': {}
        }
        
        # Análisis por algoritmo
        for algorithm in self.algorithms:
            print(f"\n🔍 Analizando {algorithm}...")
            
            phase1_data = self.data[(self.data['algorithm'] == algorithm) & 
                                   (self.data['phase'] == 1)]
            phase2_data = self.data[(self.data['algorithm'] == algorithm) & 
                                   (self.data['phase'] == 2)]
            
            if len(phase1_data) == 0 or len(phase2_data) == 0:
                print(f"  ⚠️ Datos insuficientes para {algorithm}")
                continue
            
            # Estadísticas por fase
            phase1_stats = {
                'mean_error': phase1_data['error'].mean(),
                'std_error': phase1_data['error'].std(),
                'mean_time': phase1_data['time'].mean(),
                'success_rate': (phase1_data['error'] < 5).mean() * 100,
                'runs': len(phase1_data)
            }
            
            phase2_stats = {
                'mean_error': phase2_data['error'].mean(),
                'std_error': phase2_data['error'].std(),
                'mean_time': phase2_data['time'].mean(),
                'success_rate': (phase2_data['error'] < 5).mean() * 100,
                'runs': len(phase2_data)
            }
            
            # Cálculo de degradaciones
            error_degradation = ((phase2_stats['mean_error'] - phase1_stats['mean_error']) / 
                                phase1_stats['mean_error']) * 100
            time_improvement = ((phase1_stats['mean_time'] - phase2_stats['mean_time']) / 
                              phase1_stats['mean_time']) * 100
            success_rate_change = phase2_stats['success_rate'] - phase1_stats['success_rate']
            
            # Score de adaptabilidad
            adaptability_score = max(0, 100 - abs(error_degradation) + 
                                   time_improvement * 0.2 + 
                                   success_rate_change * 0.5)
            
            # Guardar resultados
            results['phase_comparison'][algorithm] = {
                'phase1': phase1_stats,
                'phase2': phase2_stats
            }
            
            results['degradation_metrics'][algorithm] = {
                'error_degradation': error_degradation,
                'time_improvement': time_improvement,
                'success_rate_change': success_rate_change,
                'adaptability_score': adaptability_score
            }
            
            # Análisis de impacto por parámetro
            param_impact = {}
            for param in self.param_names:
                phase1_errors = phase1_data[f'error_{param}'].dropna()
                phase2_errors = phase2_data[f'error_{param}'].dropna()
                
                if len(phase1_errors) > 0 and len(phase2_errors) > 0:
                    phase1_mean = phase1_errors.mean()
                    phase2_mean = phase2_errors.mean()
                    param_degradation = ((phase2_mean - phase1_mean) / phase1_mean) * 100
                    
                    param_impact[param] = {
                        'phase1_error': phase1_mean,
                        'phase2_error': phase2_mean,
                        'degradation': param_degradation,
                        'is_problematic': param_degradation > 20
                    }
            
            results['parameter_impact'][algorithm] = param_impact
            
            print(f"  📊 Error: {phase1_stats['mean_error']:.1f}% → {phase2_stats['mean_error']:.1f}% ({error_degradation:+.1f}%)")
            print(f"  ⏱️ Tiempo: {phase1_stats['mean_time']:.1f}s → {phase2_stats['mean_time']:.1f}s ({time_improvement:+.1f}%)")
            print(f"  🎯 Score adaptabilidad: {adaptability_score:.2f}")
        
        # Ranking de adaptabilidad
        results['adaptability_ranking'] = sorted([
            {
                'algorithm': alg,
                'score': metrics['adaptability_score'],
                'error_degradation': metrics['error_degradation'],
                'time_improvement': metrics['time_improvement']
            }
            for alg, metrics in results['degradation_metrics'].items()
        ], key=lambda x: x['score'], reverse=True)
        
        # Generar insights
        best = results['adaptability_ranking'][0]
        worst = results['adaptability_ranking'][-1]
        avg_error_deg = np.mean([m['error_degradation'] for m in results['degradation_metrics'].values()])
        avg_time_imp = np.mean([m['time_improvement'] for m in results['degradation_metrics'].values()])
        
        results['insights'] = [
            f"🏆 MEJOR adaptabilidad: {best['algorithm']} (Score: {best['score']:.2f})",
            f"📉 Degradación promedio de error: {avg_error_deg:+.1f}% (multi-señal → solo corriente)",
            f"⚡ Mejora promedio en tiempo: {avg_time_imp:+.1f}% (adaptación más eficiente)",
            f"⚠️ PEOR adaptabilidad: {worst['algorithm']} (Score: {worst['score']:.2f})"
        ]
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS - Análisis de Adaptabilidad:")
        for insight in results['insights']:
            print(f"  {insight}")
        
        print(f"\n🏆 RANKING DE ADAPTABILIDAD:")
        for i, item in enumerate(results['adaptability_ranking']):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {medal} {i+1}. {item['algorithm']} (Score: {item['score']:.2f})")
        
        # Guardar resultados
        self._save_results(results, "module_1_adaptability")
        
        return results
    
    # ===== MÓDULO 2: HEATMAP DE PARÁMETROS DQ =====
    def module_2_parameter_heatmap(self):
        """
        MÓDULO 2: Heatmap de Parámetros DQ
        Análisis detallado de error por parámetro en diferentes contextos
        """
        print("\n" + "="*80)
        print("📊 MÓDULO 2: HEATMAP DE PARÁMETROS DQ")
        print("Objetivo: Error por parámetro × Algoritmo × Escenario")
        print("="*80)
        
        results = {
            'title': 'Heatmap de Parámetros DQ',
            'description': 'Análisis detallado de error por parámetro',
            'heatmap_data': {},
            'parameter_ranking': [],
            'algorithm_specialization': {},
            'scenario_impact': {},
            'observability_analysis': {},
            'insights': [],
            'summary': {}
        }
        
        # Crear matriz de datos para heatmap
        heatmap_matrix = np.zeros((len(self.algorithms), len(self.param_names)))
        
        # Análisis global por parámetro
        param_difficulty = {}
        for i, param in enumerate(self.param_names):
            all_errors = []
            
            for j, algorithm in enumerate(self.algorithms):
                alg_data = self.data[self.data['algorithm'] == algorithm]
                param_errors = alg_data[f'error_{param}'].dropna()
                
                if len(param_errors) > 0:
                    mean_error = param_errors.mean()
                    all_errors.append(mean_error)
                    heatmap_matrix[j, i] = mean_error
                else:
                    heatmap_matrix[j, i] = np.nan
            
            if all_errors:
                param_difficulty[param] = {
                    'avg_error': np.mean(all_errors),
                    'max_error': np.max(all_errors),
                    'min_error': np.min(all_errors),
                    'std_error': np.std(all_errors)
                }
        
        # Ranking de dificultad por parámetro
        results['parameter_ranking'] = sorted([
            {
                'parameter': param,
                'avg_error': data['avg_error'],
                'difficulty': 'Muy Alto' if data['avg_error'] > 25 else
                            'Alto' if data['avg_error'] > 15 else
                            'Medio' if data['avg_error'] > 10 else 'Bajo'
            }
            for param, data in param_difficulty.items()
        ], key=lambda x: x['avg_error'], reverse=True)
        
        # Especialización de algoritmos por parámetro
        for param in self.param_names:
            alg_performance = {}
            for algorithm in self.algorithms:
                alg_data = self.data[self.data['algorithm'] == algorithm]
                param_errors = alg_data[f'error_{param}'].dropna()
                if len(param_errors) > 0:
                    alg_performance[algorithm] = param_errors.mean()
            
            if alg_performance:
                best_alg = min(alg_performance.items(), key=lambda x: x[1])
                results['algorithm_specialization'][param] = {
                    'best_algorithm': best_alg[0],
                    'best_error': best_alg[1],
                    'all_performance': alg_performance
                }
        
        # Análisis de observabilidad (Fase 1 vs Fase 2)
        for param in self.param_names:
            phase1_errors = self.data[self.data['phase'] == 1][f'error_{param}'].dropna()
            phase2_errors = self.data[self.data['phase'] == 2][f'error_{param}'].dropna()
            
            if len(phase1_errors) > 0 and len(phase2_errors) > 0:
                phase1_avg = phase1_errors.mean()
                phase2_avg = phase2_errors.mean()
                observability_loss = ((phase2_avg - phase1_avg) / phase1_avg) * 100
                
                results['observability_analysis'][param] = {
                    'phase1_error': phase1_avg,
                    'phase2_error': phase2_avg,
                    'observability_loss': observability_loss,
                    'is_problematic': observability_loss > 50
                }
        
        # Generar insights
        hardest = results['parameter_ranking'][0]
        easiest = results['parameter_ranking'][-1]
        worst_observability = max(results['observability_analysis'].items(), 
                                key=lambda x: x[1]['observability_loss'])
        
        results['insights'] = [
            f"🔴 Parámetro MÁS difícil: {hardest['parameter']} ({hardest['avg_error']:.1f}% error)",
            f"🟢 Parámetro MÁS fácil: {easiest['parameter']} ({easiest['avg_error']:.1f}% error)",
            f"👁️ Mayor pérdida observabilidad: {worst_observability[0]} ({worst_observability[1]['observability_loss']:.1f}%)",
            f"📊 Rango de dificultad: {easiest['avg_error']:.1f}% - {hardest['avg_error']:.1f}%"
        ]
        
        # Crear visualización heatmap
        self._create_parameter_heatmap(heatmap_matrix, results)
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS - Heatmap de Parámetros:")
        for insight in results['insights']:
            print(f"  {insight}")
        
        print(f"\n🏆 RANKING DE DIFICULTAD:")
        for i, item in enumerate(results['parameter_ranking']):
            difficulty = item['difficulty']
            icon = '🔴' if difficulty == 'Muy Alto' else '🟠' if difficulty == 'Alto' else '🟡' if difficulty == 'Medio' else '🟢'
            print(f"  {icon} {i+1}. {item['parameter']} ({item['avg_error']:.1f}% - {difficulty})")
        
        # Guardar resultados
        self._save_results(results, "module_2_heatmap")
        
        return results
    
    # ===== MÓDULO 3: DASHBOARD COMPARATIVO =====
    def module_3_comparative_dashboard(self):
        """
        MÓDULO 3: Dashboard Comparativo
        Análisis multidimensional de rendimiento algorítmico
        """
        print("\n" + "="*80)
        print("⚡ MÓDULO 3: DASHBOARD COMPARATIVO")
        print("Objetivo: Análisis integral de Precisión + Tiempo + Robustez")
        print("="*80)
        
        results = {
            'title': 'Dashboard Comparativo Integral',
            'description': 'Análisis multidimensional de rendimiento',
            'algorithm_analysis': {},
            'overall_ranking': [],
            'tradeoff_analysis': {},
            'robustness_index': {},
            'insights': [],
            'summary': {}
        }
        
        # Análisis por algoritmo
        for algorithm in self.algorithms:
            print(f"\n🔍 Analizando {algorithm}...")
            
            alg_data = self.data[self.data['algorithm'] == algorithm]
            
            if len(alg_data) == 0:
                continue
            
            # Métricas básicas
            mean_error = alg_data['error'].mean()
            std_error = alg_data['error'].std()
            mean_time = alg_data['time'].mean()
            std_time = alg_data['time'].std()
            
            # Métricas de calidad
            success_rate = (alg_data['error'] < 5).mean() * 100
            excellence_rate = (alg_data['error'] < 2).mean() * 100
            
            # Robustez (inverso del coeficiente de variación)
            error_cv = std_error / (mean_error + 1e-8)
            time_cv = std_time / (mean_time + 1e-8)
            robustness_score = 1 / (error_cv + time_cv + 1e-8)
            
            # Eficiencia (precisión por unidad de tiempo)
            efficiency = (100 - mean_error) / (mean_time / 100)
            
            # Score general
            accuracy_score = max(0, 100 - mean_error)
            speed_score = max(0, 100 - mean_time / 10)
            robustness_norm = min(100, robustness_score * 10)
            overall_score = (accuracy_score * 0.4 + speed_score * 0.3 + robustness_norm * 0.3)
            
            results['algorithm_analysis'][algorithm] = {
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_time': mean_time,
                'std_time': std_time,
                'success_rate': success_rate,
                'excellence_rate': excellence_rate,
                'robustness_score': robustness_score,
                'efficiency': efficiency,
                'overall_score': overall_score,
                'data_points': len(alg_data)
            }
            
            print(f"  📊 Error: {mean_error:.1f}% ± {std_error:.1f}%")
            print(f"  ⏱️ Tiempo: {mean_time:.1f}s ± {std_time:.1f}s")
            print(f"  🎯 Success rate: {success_rate:.1f}%")
            print(f"  🏆 Overall score: {overall_score:.2f}")
        
        # Ranking general
        results['overall_ranking'] = sorted([
            {
                'algorithm': alg,
                'overall_score': data['overall_score'],
                'mean_error': data['mean_error'],
                'mean_time': data['mean_time'],
                'success_rate': data['success_rate']
            }
            for alg, data in results['algorithm_analysis'].items()
        ], key=lambda x: x['overall_score'], reverse=True)
        
        # Análisis de trade-offs
        results['tradeoff_analysis'] = {
            'accuracy_vs_speed': [
                {
                    'algorithm': alg,
                    'accuracy': 100 - data['mean_error'],
                    'speed': 1000 / data['mean_time'],
                    'efficiency': data['efficiency']
                }
                for alg, data in results['algorithm_analysis'].items()
            ]
        }
        
        # Robustez
        results['robustness_index'] = sorted([
            {
                'algorithm': alg,
                'robustness_score': data['robustness_score'],
                'success_rate': data['success_rate']
            }
            for alg, data in results['algorithm_analysis'].items()
        ], key=lambda x: x['robustness_score'], reverse=True)
        
        # Generar insights
        best = results['overall_ranking'][0]
        most_robust = results['robustness_index'][0]
        fastest = min(results['algorithm_analysis'].items(), key=lambda x: x[1]['mean_time'])
        
        results['insights'] = [
            f"🏆 MEJOR algoritmo general: {best['algorithm']} (Score: {best['overall_score']:.2f})",
            f"🛡️ MÁS robusto: {most_robust['algorithm']} (Robustez: {most_robust['robustness_score']:.2f})",
            f"⚡ MÁS rápido: {fastest[0]} ({fastest[1]['mean_time']:.1f}s promedio)",
            f"🎯 Mejor success rate: {best['algorithm']} ({best['success_rate']:.1f}%)"
        ]
        
        # Crear visualizaciones
        self._create_dashboard_plots(results)
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS - Dashboard Comparativo:")
        for insight in results['insights']:
            print(f"  {insight}")
        
        print(f"\n🏆 RANKING GENERAL:")
        for i, item in enumerate(results['overall_ranking']):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {medal} {i+1}. {item['algorithm']} (Score: {item['overall_score']:.2f})")
        
        # Guardar resultados
        self._save_results(results, "module_3_dashboard")
        
        return results
    
    # ===== MÓDULO 4: ANÁLISIS ESTADÍSTICO ROBUSTO =====
    def module_4_statistical_analysis(self):
        """
        MÓDULO 4: Análisis Estadístico Robusto
        ANOVA multifactorial + Tests post-hoc + Intervalos de confianza
        """
        print("\n" + "="*80)
        print("📈 MÓDULO 4: ANÁLISIS ESTADÍSTICO ROBUSTO")
        print("Objetivo: ANOVA + Post-hoc + Intervalos de confianza")
        print("="*80)
        
        results = {
            'title': 'Análisis Estadístico Robusto',
            'description': 'Análisis de significancia estadística',
            'anova_results': {},
            'posthoc_tests': {},
            'confidence_intervals': {},
            'normality_tests': {},
            'effect_sizes': {},
            'insights': [],
            'summary': {}
        }
        
        # Test de normalidad
        print("\n🔍 Tests de Normalidad:")
        for algorithm in self.algorithms:
            alg_data = self.data[self.data['algorithm'] == algorithm]['error']
            statistic, p_value = shapiro(alg_data)
            is_normal = p_value > 0.05
            
            results['normality_tests'][algorithm] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal
            }
            
            print(f"  {algorithm}: {'✓' if is_normal else '✗'} Normal (p={p_value:.4f})")
        
        # ANOVA para errores por algoritmo
        print("\n📊 ANOVA - Comparación de Errores por Algoritmo:")
        algorithm_groups = [self.data[self.data['algorithm'] == alg]['error'].values 
                          for alg in self.algorithms]
        
        f_stat, p_value = f_oneway(*algorithm_groups)
        
        results['anova_results']['algorithm_effect'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect': 'Algoritmo tiene efecto significativo' if p_value < 0.05 else 'Sin efecto significativo'
        }
        
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  {'✓ Significativo' if p_value < 0.05 else '✗ No significativo'} (α=0.05)")
        
        # ANOVA para fases
        print("\n📊 ANOVA - Comparación de Errores por Fase:")
        phase_groups = [self.data[self.data['phase'] == phase]['error'].values 
                       for phase in self.phases]
        
        f_stat_phase, p_value_phase = f_oneway(*phase_groups)
        
        results['anova_results']['phase_effect'] = {
            'f_statistic': f_stat_phase,
            'p_value': p_value_phase,
            'significant': p_value_phase < 0.05
        }
        
        print(f"  F-statistic: {f_stat_phase:.4f}")
        print(f"  P-value: {p_value_phase:.6f}")
        print(f"  {'✓ Significativo' if p_value_phase < 0.05 else '✗ No significativo'} (α=0.05)")
        
        # Tests post-hoc (pairwise t-tests)
        print("\n🔬 Tests Post-hoc (Comparaciones Pairwise):")
        posthoc_results = {}
        
        for i, alg1 in enumerate(self.algorithms):
            for j, alg2 in enumerate(self.algorithms[i+1:], i+1):
                group1 = self.data[self.data['algorithm'] == alg1]['error']
                group2 = self.data[self.data['algorithm'] == alg2]['error']
                
                t_stat, p_val = ttest_ind(group1, group2)
                
                comparison = f"{alg1}_vs_{alg2}"
                posthoc_results[comparison] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'mean_diff': group1.mean() - group2.mean()
                }
                
                print(f"  {alg1} vs {alg2}: {'✓' if p_val < 0.05 else '✗'} (p={p_val:.4f})")
        
        results['posthoc_tests'] = posthoc_results
        
        # Intervalos de confianza
        print("\n📏 Intervalos de Confianza (95%):")
        confidence_intervals = {}
        
        for algorithm in self.algorithms:
            alg_data = self.data[self.data['algorithm'] == algorithm]['error']
            mean = alg_data.mean()
            sem = stats.sem(alg_data)  # Standard Error of Mean
            ci = stats.t.interval(0.95, len(alg_data)-1, loc=mean, scale=sem)
            
            confidence_intervals[algorithm] = {
                'mean': mean,
                'lower': ci[0],
                'upper': ci[1],
                'margin_error': ci[1] - mean
            }
            
            print(f"  {algorithm}: {mean:.2f}% [{ci[0]:.2f}, {ci[1]:.2f}]")
        
        results['confidence_intervals'] = confidence_intervals
        
        # Tamaño del efecto (Cohen's d)
        print("\n📐 Tamaño del Efecto (Cohen's d):")
        effect_sizes = {}
        
        for i, alg1 in enumerate(self.algorithms):
            for j, alg2 in enumerate(self.algorithms[i+1:], i+1):
                group1 = self.data[self.data['algorithm'] == alg1]['error']
                group2 = self.data[self.data['algorithm'] == alg2]['error']
                
                # Cohen's d
                pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / 
                                   (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std
                
                comparison = f"{alg1}_vs_{alg2}"
                effect_sizes[comparison] = {
                    'cohens_d': cohens_d,
                    'magnitude': 'Grande' if abs(cohens_d) >= 0.8 else 
                               'Mediano' if abs(cohens_d) >= 0.5 else
                               'Pequeño' if abs(cohens_d) >= 0.2 else 'Negligible'
                }
                
                print(f"  {alg1} vs {alg2}: d={cohens_d:.3f} ({effect_sizes[comparison]['magnitude']})")
        
        results['effect_sizes'] = effect_sizes
        
        # Generar insights
        significant_comparisons = [comp for comp, data in posthoc_results.items() 
                                 if data['significant']]
        
        best_algorithm = min(confidence_intervals.items(), 
                           key=lambda x: x[1]['mean'])
        
        results['insights'] = [
            f"📊 Efecto algoritmo: {'Significativo' if results['anova_results']['algorithm_effect']['significant'] else 'No significativo'} (p={results['anova_results']['algorithm_effect']['p_value']:.4f})",
            f"🔄 Efecto fase: {'Significativo' if results['anova_results']['phase_effect']['significant'] else 'No significativo'} (p={results['anova_results']['phase_effect']['p_value']:.4f})",
            f"🎯 Mejor algoritmo estadísticamente: {best_algorithm[0]} ({best_algorithm[1]['mean']:.2f}% ± {best_algorithm[1]['margin_error']:.2f}%)",
            f"🔬 Comparaciones significativas: {len(significant_comparisons)}/{len(posthoc_results)}"
        ]
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS - Análisis Estadístico:")
        for insight in results['insights']:
            print(f"  {insight}")
        
        # Guardar resultados
        self._save_results(results, "module_4_statistical")
        
        return results
    
    # ===== MÓDULO 5: ANÁLISIS DE CONVERGENCIA =====
    def module_5_convergence_analysis(self):
        """
        MÓDULO 5: Análisis de Convergencia
        Eficiencia de convergencia y estabilidad
        """
        print("\n" + "="*80)
        print("🎯 MÓDULO 5: ANÁLISIS DE CONVERGENCIA")
        print("Objetivo: Eficiencia temporal y estabilidad")
        print("="*80)
        
        results = {
            'title': 'Análisis de Convergencia',
            'description': 'Eficiencia de convergencia y estabilidad temporal',
            'convergence_metrics': {},
            'efficiency_analysis': {},
            'stability_metrics': {},
            'cost_analysis': {},
            'insights': [],
            'summary': {}
        }
        
        # Análisis por algoritmo
        for algorithm in self.algorithms:
            print(f"\n🔍 Analizando convergencia de {algorithm}...")
            
            alg_data = self.data[self.data['algorithm'] == algorithm]
            
            if len(alg_data) == 0:
                continue
            
            # Métricas de convergencia
            times = alg_data['time'].values
            errors = alg_data['error'].values
            costs = alg_data['cost'].values
            
            # Velocidad de convergencia (error/tiempo)
            convergence_speed = np.mean(errors / times)
            
            # Eficiencia (1 - error) / tiempo
            efficiency_scores = (100 - errors) / times * 100
            mean_efficiency = np.mean(efficiency_scores)
            
            # Estabilidad temporal
            time_cv = np.std(times) / (np.mean(times) + 1e-8)
            error_cv = np.std(errors) / (np.mean(errors) + 1e-8)
            cost_cv = np.std(costs) / (np.mean(costs) + 1e-8)
            
            stability_index = 1 / (time_cv + error_cv + cost_cv + 1e-8)
            
            # Análisis de costo vs beneficio
            cost_effectiveness = np.mean((100 - errors) / (costs + 1e-8))
            
            # Análisis por fase
            phase_analysis = {}
            for phase in self.phases:
                phase_data = alg_data[alg_data['phase'] == phase]
                if len(phase_data) > 0:
                    phase_analysis[f'phase_{phase}'] = {
                        'mean_time': phase_data['time'].mean(),
                        'mean_error': phase_data['error'].mean(),
                        'mean_cost': phase_data['cost'].mean(),
                        'efficiency': np.mean((100 - phase_data['error']) / phase_data['time'] * 100)
                    }
            
            results['convergence_metrics'][algorithm] = {
                'convergence_speed': convergence_speed,
                'mean_efficiency': mean_efficiency,
                'stability_index': stability_index,
                'time_cv': time_cv,
                'error_cv': error_cv,
                'cost_cv': cost_cv,
                'cost_effectiveness': cost_effectiveness,
                'phase_analysis': phase_analysis
            }
            
            print(f"  📈 Velocidad convergencia: {convergence_speed:.4f}")
            print(f"  ⚡ Eficiencia promedio: {mean_efficiency:.2f}")
            print(f"  🛡️ Índice estabilidad: {stability_index:.2f}")
            print(f"  💰 Costo-efectividad: {cost_effectiveness:.2f}")
        
        # Ranking de eficiencia
        efficiency_ranking = sorted([
            {
                'algorithm': alg,
                'efficiency': data['mean_efficiency'],
                'stability': data['stability_index'],
                'cost_effectiveness': data['cost_effectiveness']
            }
            for alg, data in results['convergence_metrics'].items()
        ], key=lambda x: x['efficiency'], reverse=True)
        
        results['efficiency_analysis'] = {
            'ranking': efficiency_ranking,
            'best_efficiency': efficiency_ranking[0] if efficiency_ranking else None,
            'most_stable': max(results['convergence_metrics'].items(), 
                             key=lambda x: x[1]['stability_index']) if results['convergence_metrics'] else None,
            'most_cost_effective': max(results['convergence_metrics'].items(), 
                                     key=lambda x: x[1]['cost_effectiveness']) if results['convergence_metrics'] else None
        }
        
        # Análisis de estabilidad comparativa
        stability_comparison = {}
        for metric in ['time_cv', 'error_cv', 'cost_cv']:
            stability_comparison[metric] = sorted([
                {'algorithm': alg, 'cv': data[metric]}
                for alg, data in results['convergence_metrics'].items()
            ], key=lambda x: x['cv'])
        
        results['stability_metrics'] = stability_comparison
        
        # Generar insights
        best_efficiency = results['efficiency_analysis']['best_efficiency']
        most_stable = results['efficiency_analysis']['most_stable']
        most_cost_effective = results['efficiency_analysis']['most_cost_effective']
        
        results['insights'] = [
            f"⚡ MÁS eficiente: {best_efficiency['algorithm']} ({best_efficiency['efficiency']:.2f})" if best_efficiency else "Sin datos de eficiencia",
            f"🛡️ MÁS estable: {most_stable[0]} (Estabilidad: {most_stable[1]['stability_index']:.2f})" if most_stable else "Sin datos de estabilidad",
            f"💰 MÁS costo-efectivo: {most_cost_effective[0]} ({most_cost_effective[1]['cost_effectiveness']:.2f})" if most_cost_effective else "Sin datos de costo",
            f"📊 Rango eficiencia: {efficiency_ranking[-1]['efficiency']:.1f} - {efficiency_ranking[0]['efficiency']:.1f}" if len(efficiency_ranking) > 1 else "Rango único"
        ]
        
        # Crear visualizaciones
        self._create_convergence_plots(results)
        
        # Imprimir resultados
        print(f"\n📊 RESULTADOS - Análisis de Convergencia:")
        for insight in results['insights']:
            print(f"  {insight}")
        
        print(f"\n🏆 RANKING DE EFICIENCIA:")
        for i, item in enumerate(efficiency_ranking):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"  {medal} {i+1}. {item['algorithm']} (Eficiencia: {item['efficiency']:.2f})")
        
        # Guardar resultados
        self._save_results(results, "module_5_convergence")
        
        return results
    
    # ===== MÉTODOS DE VISUALIZACIÓN =====
    def _create_parameter_heatmap(self, heatmap_matrix, results):
        """Crear heatmap de parámetros DQ"""
        plt.figure(figsize=(12, 8))
        
        # Crear heatmap
        mask = np.isnan(heatmap_matrix)
        sns.heatmap(heatmap_matrix, 
                   xticklabels=self.param_names,
                   yticklabels=self.algorithms,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlBu_r',
                   mask=mask,
                   cbar_kws={'label': 'Error Promedio (%)'})
        
        plt.title('Heatmap de Error por Parámetro DQ y Algoritmo', fontsize=16, fontweight='bold')
        plt.xlabel('Parámetros DQ', fontweight='bold')
        plt.ylabel('Algoritmos', fontweight='bold')
        plt.tight_layout()
        
        # Guardar
        plt.savefig(self.results_path / "plots" / "parameter_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  📊 Heatmap guardado: parameter_heatmap.png")
    
    def _create_dashboard_plots(self, results):
        """Crear visualizaciones del dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Ranking general
        algorithms = [item['algorithm'] for item in results['overall_ranking']]
        scores = [item['overall_score'] for item in results['overall_ranking']]
        
        bars = axes[0,0].bar(algorithms, scores, color=['gold', 'silver', '#CD7F32'])
        axes[0,0].set_title('Ranking General de Algoritmos', fontweight='bold')
        axes[0,0].set_ylabel('Score General')
        axes[0,0].set_ylim(0, max(scores) * 1.1)
        
        # Añadir valores en las barras
        for bar, score in zip(bars, scores):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Error vs Tiempo
        errors = [item['mean_error'] for item in results['overall_ranking']]
        times = [item['mean_time'] for item in results['overall_ranking']]
        
        scatter = axes[0,1].scatter(times, errors, c=scores, cmap='viridis', s=100, alpha=0.7)
        axes[0,1].set_xlabel('Tiempo Promedio (s)')
        axes[0,1].set_ylabel('Error Promedio (%)')
        axes[0,1].set_title('Trade-off Error vs Tiempo', fontweight='bold')
        
        # Añadir etiquetas
        for i, alg in enumerate(algorithms):
            axes[0,1].annotate(alg, (times[i], errors[i]), 
                              xytext=(5, 5), textcoords='offset points')
        
        plt.colorbar(scatter, ax=axes[0,1], label='Score General')
        
        # Plot 3: Success Rate
        success_rates = [item['success_rate'] for item in results['overall_ranking']]
        
        bars = axes[1,0].bar(algorithms, success_rates, color=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1,0].set_title('Tasa de Éxito por Algoritmo', fontweight='bold')
        axes[1,0].set_ylabel('Success Rate (%)')
        axes[1,0].set_ylim(0, 100)
        
        # Plot 4: Robustez
        robustness_data = results['robustness_index']
        alg_rob = [item['algorithm'] for item in robustness_data]
        rob_scores = [item['robustness_score'] for item in robustness_data]
        
        bars = axes[1,1].bar(alg_rob, rob_scores, color=['orange', 'purple', 'brown'])
        axes[1,1].set_title('Índice de Robustez', fontweight='bold')
        axes[1,1].set_ylabel('Score de Robustez')
        
        plt.tight_layout()
        plt.savefig(self.results_path / "plots" / "dashboard_comparative.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  📊 Dashboard guardado: dashboard_comparative.png")
    
    def _create_convergence_plots(self, results):
        """Crear visualizaciones de convergencia"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(results['convergence_metrics'].keys())
        
        # Plot 1: Eficiencia
        efficiencies = [results['convergence_metrics'][alg]['mean_efficiency'] for alg in algorithms]
        bars = axes[0,0].bar(algorithms, efficiencies, color=['cyan', 'magenta', 'yellow'])
        axes[0,0].set_title('Eficiencia de Convergencia', fontweight='bold')
        axes[0,0].set_ylabel('Eficiencia')
        
        # Plot 2: Estabilidad
        stabilities = [results['convergence_metrics'][alg]['stability_index'] for alg in algorithms]
        bars = axes[0,1].bar(algorithms, stabilities, color=['lightsteelblue', 'lightpink', 'lightyellow'])
        axes[0,1].set_title('Índice de Estabilidad', fontweight='bold')
        axes[0,1].set_ylabel('Estabilidad')
        
        # Plot 3: Costo-Efectividad
        cost_eff = [results['convergence_metrics'][alg]['cost_effectiveness'] for alg in algorithms]
        bars = axes[1,0].bar(algorithms, cost_eff, color=['lightseagreen', 'lightsalmon', 'lightgoldenrodyellow'])
        axes[1,0].set_title('Costo-Efectividad', fontweight='bold')
        axes[1,0].set_ylabel('Costo-Efectividad')
        
        # Plot 4: Coeficientes de Variación
        time_cvs = [results['convergence_metrics'][alg]['time_cv'] for alg in algorithms]
        error_cvs = [results['convergence_metrics'][alg]['error_cv'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = axes[1,1].bar(x - width/2, time_cvs, width, label='Tiempo CV', alpha=0.8)
        bars2 = axes[1,1].bar(x + width/2, error_cvs, width, label='Error CV', alpha=0.8)
        
        axes[1,1].set_title('Variabilidad (Coeficiente de Variación)', fontweight='bold')
        axes[1,1].set_ylabel('CV')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(algorithms)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_path / "plots" / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  📊 Convergencia guardado: convergence_analysis.png")
    
    # ===== MÉTODOS AUXILIARES =====
    def _save_results(self, results, module_name):
        """Guardar resultados en JSON"""
        # Convertir numpy arrays a listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_numpy(results)
        
        with open(self.results_path / f"{module_name}_results.json", 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"  💾 Resultados guardados: {module_name}_results.json")
    
    def run_all_modules(self):
        """Ejecutar todos los módulos de análisis"""
        print("\n" + "🚀" + "="*78 + "🚀")
        print("🔬 EJECUTANDO SISTEMA COMPLETO DE ANÁLISIS")
        print("🚀" + "="*78 + "🚀")
        
        all_results = {}
        
        # Ejecutar todos los módulos
        all_results['module_1'] = self.module_1_adaptability_analysis()
        all_results['module_2'] = self.module_2_parameter_heatmap()
        all_results['module_3'] = self.module_3_comparative_dashboard()
        all_results['module_4'] = self.module_4_statistical_analysis()
        all_results['module_5'] = self.module_5_convergence_analysis()
        
        # Generar resumen ejecutivo
        self._generate_executive_summary(all_results)
        
        print("\n" + "✅" + "="*78 + "✅")
        print("🎯 ANÁLISIS COMPLETO FINALIZADO")
        print(f"📁 Resultados guardados en: {self.results_path}")
        print("✅" + "="*78 + "✅")
        
        return all_results
    
    def _generate_executive_summary(self, all_results):
        """Generar resumen ejecutivo de todos los módulos"""
        summary = {
            'title': 'Resumen Ejecutivo - Sistema de Gemelo Digital Adaptativo',
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key_findings': [],
            'recommendations': [],
            'best_algorithms': {},
            'critical_parameters': [],
            'statistical_significance': {}
        }
        
        # Extraer hallazgos clave de cada módulo
        if 'module_1' in all_results:
            best_adaptable = all_results['module_1']['adaptability_ranking'][0]
            summary['best_algorithms']['adaptability'] = best_adaptable['algorithm']
            summary['key_findings'].append(f"Mejor adaptabilidad: {best_adaptable['algorithm']}")
        
        if 'module_2' in all_results:
            hardest_param = all_results['module_2']['parameter_ranking'][0]
            summary['critical_parameters'].append(hardest_param['parameter'])
            summary['key_findings'].append(f"Parámetro más crítico: {hardest_param['parameter']}")
        
        if 'module_3' in all_results:
            best_overall = all_results['module_3']['overall_ranking'][0]
            summary['best_algorithms']['overall'] = best_overall['algorithm']
            summary['key_findings'].append(f"Mejor rendimiento general: {best_overall['algorithm']}")
        
        if 'module_4' in all_results:
            is_significant = all_results['module_4']['anova_results']['algorithm_effect']['significant']
            summary['statistical_significance']['algorithm_effect'] = is_significant
            summary['key_findings'].append(f"Diferencias estadísticamente significativas: {'Sí' if is_significant else 'No'}")
        
        # Recomendaciones
        summary['recommendations'] = [
            f"Para implementación industrial: Usar {summary['best_algorithms'].get('overall', 'N/A')}",
            f"Para máxima adaptabilidad: Usar {summary['best_algorithms'].get('adaptability', 'N/A')}",
            f"Monitorear especialmente el parámetro: {summary['critical_parameters'][0] if summary['critical_parameters'] else 'N/A'}",
            "Implementar sistema de 2 fases: Calibración completa + Adaptación con corriente",
            "Considerar robustez vs precisión según aplicación específica"
        ]
        
        # Guardar resumen
        with open(self.results_path / "executive_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Imprimir resumen
        print("\n" + "📋" + "="*78 + "📋")
        print("📊 RESUMEN EJECUTIVO")
        print("📋" + "="*78 + "📋")
        
        print("\n🎯 HALLAZGOS CLAVE:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        print("\n💡 RECOMENDACIONES:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n💾 Resumen guardado: executive_summary.json")


# ===== FUNCIÓN PRINCIPAL =====
def main():
    """Función principal para ejecutar el análisis"""
    print("🔬 SISTEMA MODULAR DE ANÁLISIS")
    print("Gemelo Digital Adaptativo - Algoritmos Bio-Inspirados")
    print("="*80)
    
    # Inicializar analizador
    analyzer = AdaptiveDigitalTwinAnalyzer(data_path="D:\\GitHub\\M_Sc\\TESIS\\Congreso\\CIMCIA\\Final\\Test_13082025_confg_optimizada\\results\\csv\\")

    # Ejecutar análisis completo
    results = analyzer.run_all_modules()
    
    print("\n🎉 ¡Análisis completado exitosamente!")
    print("📁 Revisa la carpeta 'results_analysis' para todos los resultados")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
