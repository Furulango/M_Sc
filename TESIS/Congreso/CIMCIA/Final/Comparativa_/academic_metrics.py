import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def friedman_test_with_posthoc(df, metrics, algorithms):
    """Friedman test for multiple algorithms with post-hoc analysis"""
    results = {}
    
    for metric in metrics:
        # Prepare data matrix (algorithms x runs)
        data_matrix = []
        valid_algorithms = []
        
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg][metric].dropna().values
            if len(alg_data) > 0:
                # Pad or truncate to same length
                min_length = min([len(df[df['Algorithm'] == a][metric].dropna()) for a in algorithms])
                data_matrix.append(alg_data[:min_length])
                valid_algorithms.append(alg)
        
        if len(data_matrix) >= 3 and len(data_matrix[0]) >= 5:
            # Friedman test
            statistic, p_value = friedmanchisquare(*data_matrix)
            
            # Post-hoc Wilcoxon signed-rank tests with Bonferroni correction
            n_comparisons = len(valid_algorithms) * (len(valid_algorithms) - 1) // 2
            alpha_corrected = 0.05 / n_comparisons
            
            posthoc_results = {}
            for i, alg1 in enumerate(valid_algorithms):
                for j, alg2 in enumerate(valid_algorithms):
                    if i < j:
                        stat, p = wilcoxon(data_matrix[i], data_matrix[j], alternative='two-sided')
                        posthoc_results[f'{alg1}_vs_{alg2}'] = {
                            'statistic': stat,
                            'p_value': p,
                            'significant_bonferroni': p < alpha_corrected,
                            'significant_uncorrected': p < 0.05
                        }
            
            results[metric] = {
                'friedman_statistic': statistic,
                'friedman_p_value': p_value,
                'significant': p_value < 0.05,
                'posthoc_comparisons': posthoc_results,
                'bonferroni_alpha': alpha_corrected
            }
    
    return results

def calculate_effect_sizes(df, metrics, algorithms):
    """Calculate Cohen's d and other effect sizes between algorithms"""
    results = {}
    
    for metric in metrics:
        metric_results = {}
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    data1 = df[df['Algorithm'] == alg1][metric].dropna().values
                    data2 = df[df['Algorithm'] == alg2][metric].dropna().values
                    
                    if len(data1) > 1 and len(data2) > 1:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + 
                                            (len(data2)-1)*np.var(data2, ddof=1)) / 
                                           (len(data1) + len(data2) - 2))
                        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                        
                        # Glass's delta
                        glass_delta = (np.mean(data1) - np.mean(data2)) / np.std(data2, ddof=1) if np.std(data2, ddof=1) > 0 else 0
                        
                        # Hedges' g (bias-corrected Cohen's d)
                        df_total = len(data1) + len(data2) - 2
                        correction = 1 - (3 / (4 * df_total - 1))
                        hedges_g = cohens_d * correction
                        
                        # Cliff's delta (non-parametric effect size)
                        cliffs_delta = cliff_delta(data1, data2)
                        
                        metric_results[f'{alg1}_vs_{alg2}'] = {
                            'cohens_d': cohens_d,
                            'glass_delta': glass_delta,
                            'hedges_g': hedges_g,
                            'cliffs_delta': cliffs_delta,
                            'effect_size_interpretation': interpret_effect_size(abs(cohens_d))
                        }
        
        results[metric] = metric_results
    
    return results

def cliff_delta(x, y):
    """Calculate Cliff's delta effect size"""
    n1, n2 = len(x), len(y)
    delta = 0
    
    for i in range(n1):
        for j in range(n2):
            if x[i] > y[j]:
                delta += 1
            elif x[i] < y[j]:
                delta -= 1
    
    return delta / (n1 * n2)

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def advanced_ranking_analysis(df, metrics, algorithms):
    """Advanced ranking analysis using multiple criteria"""
    results = {}
    
    # Rank-based analysis
    ranking_matrix = np.zeros((len(algorithms), len(metrics)))
    
    for j, metric in enumerate(metrics):
        metric_values = {}
        for alg in algorithms:
            values = df[df['Algorithm'] == alg][metric].dropna().values
            if len(values) > 0:
                metric_values[alg] = np.mean(values)
        
        # Rank algorithms for this metric (1 = best)
        if 'Error' in metric or 'Cost' in metric or 'Time' in metric:
            # Lower is better
            sorted_algs = sorted(metric_values.items(), key=lambda x: x[1])
        else:
            # Higher is better
            sorted_algs = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (alg, value) in enumerate(sorted_algs, 1):
            alg_idx = algorithms.index(alg)
            ranking_matrix[alg_idx, j] = rank
    
    # Calculate average ranks
    average_ranks = np.mean(ranking_matrix, axis=1)
    
    # Borda count (sum of ranks, lower is better)
    borda_scores = np.sum(ranking_matrix, axis=1)
    
    # Normalized Borda count
    normalized_borda = (np.max(borda_scores) - borda_scores) / (np.max(borda_scores) - np.min(borda_scores))
    
    results = {
        'ranking_matrix': ranking_matrix,
        'average_ranks': {alg: avg_rank for alg, avg_rank in zip(algorithms, average_ranks)},
        'borda_scores': {alg: score for alg, score in zip(algorithms, borda_scores)},
        'normalized_borda': {alg: norm_score for alg, norm_score in zip(algorithms, normalized_borda)},
        'overall_ranking': sorted(zip(algorithms, average_ranks), key=lambda x: x[1])
    }
    
    return results

def performance_stability_analysis(df, algorithms):
    """Analyze performance stability across runs"""
    results = {}
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        
        if len(alg_data) > 5:  # Need sufficient data
            # Stability metrics
            error_values = alg_data['Average_Error_Pct'].values
            
            # Quartile coefficient of dispersion
            q1, q3 = np.percentile(error_values, [25, 75])
            qcd = (q3 - q1) / (q3 + q1) if (q3 + q1) > 0 else 0
            
            # Relative standard deviation
            rsd = np.std(error_values, ddof=1) / np.mean(error_values) * 100 if np.mean(error_values) > 0 else 0
            
            # Success rate stability (variance in success across metrics)
            success_metrics = ['Criterion_Error10pct', 'Criterion_Cost001', 'Criterion_5Params5pct', 'Criterion_R2_95pct']
            success_rates = []
            for metric in success_metrics:
                if metric in alg_data.columns:
                    success_rates.append(alg_data[metric].mean())
            
            success_stability = 1 - np.std(success_rates) if len(success_rates) > 1 else 1
            
            # Worst-case analysis
            worst_10pct = np.percentile(error_values, 90)  # 90th percentile
            best_10pct = np.percentile(error_values, 10)   # 10th percentile
            worst_to_best_ratio = worst_10pct / best_10pct if best_10pct > 0 else np.inf
            
            results[alg] = {
                'quartile_coeff_dispersion': qcd,
                'relative_std_dev': rsd,
                'success_rate_stability': success_stability,
                'worst_10pct_error': worst_10pct,
                'best_10pct_error': best_10pct,
                'worst_to_best_ratio': worst_to_best_ratio,
                'stability_score': (1-qcd) * success_stability * (1/(1+rsd/100))
            }
    
    return results

def convergence_quality_analysis(df, algorithms):
    """Analyze quality of convergence beyond just success rate"""
    results = {}
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        
        # Convergence speed proxy (assuming lower cost indicates faster convergence)
        final_costs = alg_data['Final_Cost'].values
        
        # Quality metrics
        convergence_consistency = 1 - (np.std(final_costs, ddof=1) / np.mean(final_costs)) if np.mean(final_costs) > 0 else 0
        
        # Multi-objective convergence (how many criteria typically met)
        criteria_met = alg_data['Criteria_Met'].values if 'Criteria_Met' in alg_data.columns else np.zeros(len(alg_data))
        avg_criteria_met = np.mean(criteria_met)
        criteria_consistency = 1 - (np.std(criteria_met, ddof=1) / 4)  # 4 is max criteria
        
        # Convergence reliability (percentage of runs meeting at least 2 criteria)
        reliability = np.mean(criteria_met >= 2)
        
        # Premium convergence (percentage meeting all 4 criteria)
        premium_rate = np.mean(criteria_met == 4)
        
        results[alg] = {
            'convergence_consistency': max(0, convergence_consistency),
            'avg_criteria_met': avg_criteria_met,
            'criteria_consistency': max(0, criteria_consistency),
            'reliability_rate': reliability,
            'premium_convergence_rate': premium_rate,
            'convergence_quality_score': (convergence_consistency + criteria_consistency + reliability) / 3
        }
    
    return results

def create_academic_summary_table(df, algorithms):
    """Create comprehensive academic summary table"""
    
    # Key metrics for academic reporting
    key_metrics = ['Average_Error_Pct', 'Time_s', 'Final_Cost', 'R_Squared']
    
    summary_data = []
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        
        row = {'Algorithm': alg}
        
        for metric in key_metrics:
            values = alg_data[metric].dropna().values
            if len(values) > 0:
                row[f'{metric}_Mean'] = np.mean(values)
                row[f'{metric}_Std'] = np.std(values, ddof=1)
                row[f'{metric}_Median'] = np.median(values)
                row[f'{metric}_IQR'] = np.percentile(values, 75) - np.percentile(values, 25)
                row[f'{metric}_Min'] = np.min(values)
                row[f'{metric}_Max'] = np.max(values)
        
        # Success metrics
        row['Success_Rate'] = alg_data['Converged_Successfully'].mean() if 'Converged_Successfully' in alg_data.columns else 0
        row['Avg_Criteria_Met'] = alg_data['Criteria_Met'].mean() if 'Criteria_Met' in alg_data.columns else 0
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def run_academic_analysis(filepath):
    """Run complete academic-focused analysis"""
    # Load data
    df = pd.read_csv(filepath)
    df_clean = df[~df['Algorithm'].str.contains('#|Dataset', na=False)].dropna(subset=['Algorithm'])
    
    algorithms = df_clean['Algorithm'].unique().tolist()
    key_metrics = ['Average_Error_Pct', 'Time_s', 'Final_Cost', 'R_Squared']
    
    print("Running academic statistical analysis...")
    
    # Run all academic analyses
    friedman_results = friedman_test_with_posthoc(df_clean, key_metrics, algorithms)
    effect_sizes = calculate_effect_sizes(df_clean, key_metrics, algorithms)
    ranking_analysis = advanced_ranking_analysis(df_clean, key_metrics, algorithms)
    stability_analysis = performance_stability_analysis(df_clean, algorithms)
    convergence_quality = convergence_quality_analysis(df_clean, algorithms)
    summary_table = create_academic_summary_table(df_clean, algorithms)
    
    results = {
        'friedman_tests': friedman_results,
        'effect_sizes': effect_sizes,
        'ranking_analysis': ranking_analysis,
        'stability_analysis': stability_analysis,
        'convergence_quality': convergence_quality,
        'summary_table': summary_table,
        'algorithms': algorithms,
        'metrics': key_metrics
    }
    
    print("Academic analysis completed.")
    return results

def print_academic_results(results):
    """Print formatted academic results"""
    
    print("\n" + "="*80)
    print("ACADEMIC STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    # Algorithm ranking
    print("\nOVERALL ALGORITHM RANKING (by average rank):")
    print("-" * 50)
    for rank, (alg, avg_rank) in enumerate(results['ranking_analysis']['overall_ranking'], 1):
        print(f"{rank}. {alg}: {avg_rank:.2f}")
    
    # Friedman test results
    print("\nFRIEDMAN TEST RESULTS (Non-parametric ANOVA):")
    print("-" * 50)
    for metric, test_result in results['friedman_tests'].items():
        sig_marker = "***" if test_result['significant'] else ""
        print(f"{metric}: χ²={test_result['friedman_statistic']:.3f}, p={test_result['friedman_p_value']:.4f} {sig_marker}")
    
    # Effect sizes summary
    print("\nEFFECT SIZES (Cohen's d - Large effects |d| > 0.8):")
    print("-" * 50)
    for metric, comparisons in results['effect_sizes'].items():
        print(f"\n{metric}:")
        for comparison, effect_data in comparisons.items():
            d = effect_data['cohens_d']
            interpretation = effect_data['effect_size_interpretation']
            if abs(d) > 0.5:  # Only show medium+ effects
                print(f"  {comparison.replace('_vs_', ' vs ')}: d={d:.3f} ({interpretation})")
    
    # Stability analysis
    print("\nSTABILITY ANALYSIS:")
    print("-" * 50)
    for alg, stability_data in results['stability_analysis'].items():
        score = stability_data['stability_score']
        rsd = stability_data['relative_std_dev']
        print(f"{alg}: Stability={score:.3f}, RSD={rsd:.1f}%")
    
    # Convergence quality
    print("\nCONVERGENCE QUALITY:")
    print("-" * 50)
    for alg, conv_data in results['convergence_quality'].items():
        quality = conv_data['convergence_quality_score']
        reliability = conv_data['reliability_rate']
        premium = conv_data['premium_convergence_rate']
        print(f"{alg}: Quality={quality:.3f}, Reliability={reliability:.1%}, Premium={premium:.1%}")

if __name__ == "__main__":
    results = run_academic_analysis("parameter_identification_results.csv")
    print_academic_results(results)