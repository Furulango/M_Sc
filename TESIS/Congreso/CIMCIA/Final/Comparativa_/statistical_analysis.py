import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, shapiro, levene
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    df_clean = df[~df['Algorithm'].str.contains('#|Dataset', na=False)]
    df_clean = df_clean.dropna(subset=['Algorithm'])
    
    boolean_cols = ['Converged_Successfully', 'Criterion_Error10pct', 'Criterion_Cost001', 
                   'Criterion_5Params5pct', 'Criterion_R2_95pct']
    for col in boolean_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.lower().map({'true': True, 'false': False})
    
    return df_clean

def statistical_significance_tests(df, metrics):
    results = {}
    algorithms = df['Algorithm'].unique()
    
    for metric in metrics:
        metric_data = [df[df['Algorithm'] == alg][metric].dropna().values for alg in algorithms]
        
        # Normality tests
        normal_tests = {}
        for i, alg in enumerate(algorithms):
            if len(metric_data[i]) >= 3:
                stat, p = shapiro(metric_data[i])
                normal_tests[alg] = {'statistic': stat, 'p_value': p, 'is_normal': p > 0.05}
        
        # Homogeneity of variance
        if len([data for data in metric_data if len(data) >= 3]) >= 2:
            levene_stat, levene_p = levene(*[data for data in metric_data if len(data) >= 3])
            equal_var = levene_p > 0.05
        else:
            equal_var = True
            levene_stat, levene_p = np.nan, np.nan
        
        # Choose appropriate test
        all_normal = all([test['is_normal'] for test in normal_tests.values()])
        
        if all_normal and equal_var and len(algorithms) > 2:
            f_stat, p_value = stats.f_oneway(*metric_data)
            test_used = 'ANOVA'
        else:
            h_stat, p_value = kruskal(*metric_data)
            test_used = 'Kruskal-Wallis'
            f_stat = h_stat
        
        # Post-hoc pairwise comparisons
        pairwise = {}
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    data1, data2 = metric_data[i], metric_data[j]
                    if len(data1) > 0 and len(data2) > 0:
                        u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                        effect_size = 1 - (2 * u_stat) / (len(data1) * len(data2))
                        pairwise[f'{alg1}_vs_{alg2}'] = {
                            'u_statistic': u_stat, 'p_value': p_val, 'effect_size': effect_size,
                            'significant': p_val < 0.05
                        }
        
        results[metric] = {
            'normality_tests': normal_tests,
            'levene_test': {'statistic': levene_stat, 'p_value': levene_p, 'equal_variance': equal_var},
            'omnibus_test': {'test': test_used, 'statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05},
            'pairwise_comparisons': pairwise
        }
    
    return results

def robustness_analysis(df, metrics):
    results = {}
    algorithms = df['Algorithm'].unique()
    
    for metric in metrics:
        metric_results = {}
        for alg in algorithms:
            data = df[df['Algorithm'] == alg][metric].dropna()
            
            if len(data) > 0:
                metric_results[alg] = {
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),
                    'cv': np.std(data, ddof=1) / np.mean(data) * 100 if np.mean(data) != 0 else np.inf,
                    'q25': np.percentile(data, 25),
                    'median': np.median(data),
                    'q75': np.percentile(data, 75),
                    'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                    'p95': np.percentile(data, 95),
                    'min': np.min(data),
                    'max': np.max(data),
                    'range': np.max(data) - np.min(data),
                    'mad': np.median(np.abs(data - np.median(data))),
                    'robustness_index': 1 / (1 + np.std(data, ddof=1) / np.mean(data)) if np.mean(data) != 0 else 0
                }
        
        results[metric] = metric_results
    
    return results

def multivariate_analysis(df, metrics):
    algorithms = df['Algorithm'].unique()
    
    # Prepare data matrix
    algorithm_means = []
    algorithm_labels = []
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        means = [alg_data[metric].mean() for metric in metrics]
        algorithm_means.append(means)
        algorithm_labels.append(alg)
    
    X = np.array(algorithm_means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Correlation matrix
    correlation_matrix = df[metrics].corr()
    
    # Composite ranking
    df_normalized = df.copy()
    for metric in metrics:
        if df[metric].std() != 0:
            if 'Error' in metric or 'MSE' in metric or 'MAE' in metric or 'Time' in metric:
                df_normalized[f'{metric}_norm'] = (df[metric].max() - df[metric]) / (df[metric].max() - df[metric].min())
            else:
                df_normalized[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    normalized_cols = [f'{metric}_norm' for metric in metrics if f'{metric}_norm' in df_normalized.columns]
    df_normalized['composite_score'] = df_normalized[normalized_cols].mean(axis=1)
    
    composite_ranking = df_normalized.groupby('Algorithm')['composite_score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    
    return {
        'pca': {
            'components': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'transformed_data': X_pca,
            'feature_names': metrics,
            'algorithm_labels': algorithm_labels
        },
        'correlation_matrix': correlation_matrix,
        'composite_ranking': composite_ranking,
        'normalized_scores': df_normalized[['Algorithm'] + normalized_cols + ['composite_score']]
    }

def efficiency_analysis(df):
    algorithms = df['Algorithm'].unique()
    results = {}
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        
        # Pareto efficiency: Error vs Time
        error_metric = 'Average_Error_Pct'
        time_metric = 'Time_s'
        
        errors = alg_data[error_metric].values
        times = alg_data[time_metric].values
        
        # Normalize for fair comparison
        norm_errors = (errors - errors.min()) / (errors.max() - errors.min()) if errors.max() != errors.min() else np.zeros_like(errors)
        norm_times = (times - times.min()) / (times.max() - times.min()) if times.max() != times.min() else np.zeros_like(times)
        
        # Efficiency index (lower is better)
        efficiency_index = norm_errors + norm_times
        best_efficiency_idx = np.argmin(efficiency_index)
        
        results[alg] = {
            'mean_error': np.mean(errors),
            'mean_time': np.mean(times),
            'efficiency_index': np.mean(efficiency_index),
            'best_run': {
                'error': errors[best_efficiency_idx],
                'time': times[best_efficiency_idx],
                'efficiency': efficiency_index[best_efficiency_idx]
            },
            'error_time_ratio': np.mean(errors) / np.mean(times),
            'normalized_performance': 1 / (1 + np.mean(efficiency_index))
        }
    
    return results

def convergence_analysis(df):
    algorithms = df['Algorithm'].unique()
    results = {}
    
    criteria_cols = ['Criterion_Error10pct', 'Criterion_Cost001', 'Criterion_5Params5pct', 'Criterion_R2_95pct']
    
    for alg in algorithms:
        alg_data = df[df['Algorithm'] == alg]
        
        success_rates = {}
        for criterion in criteria_cols:
            if criterion in alg_data.columns:
                success_rate = alg_data[criterion].mean()
                success_rates[criterion] = success_rate
        
        overall_success = alg_data['Converged_Successfully'].mean() if 'Converged_Successfully' in alg_data.columns else 0
        
        # Probability analysis
        criteria_met = alg_data['Criteria_Met'].values if 'Criteria_Met' in alg_data.columns else np.zeros(len(alg_data))
        
        results[alg] = {
            'overall_success_rate': overall_success,
            'individual_criteria_success': success_rates,
            'mean_criteria_met': np.mean(criteria_met),
            'probability_all_criteria': np.mean(criteria_met == 4),
            'probability_most_criteria': np.mean(criteria_met >= 3),
            'probability_some_criteria': np.mean(criteria_met >= 2),
            'convergence_distribution': np.bincount(criteria_met.astype(int), minlength=5) / len(criteria_met)
        }
    
    return results

def parameter_specific_analysis(df):
    param_cols = ['Error_pct_rs', 'Error_pct_rr', 'Error_pct_Lls', 'Error_pct_Llr', 'Error_pct_Lm', 'Error_pct_J', 'Error_pct_B']
    param_names = ['rs', 'rr', 'Lls', 'Llr', 'Lm', 'J', 'B']
    algorithms = df['Algorithm'].unique()
    
    results = {}
    
    for param, param_name in zip(param_cols, param_names):
        param_results = {}
        
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            errors = alg_data[param].dropna().values
            
            if len(errors) > 0:
                param_results[alg] = {
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors, ddof=1),
                    'success_rate_5pct': np.mean(errors < 5.0),
                    'success_rate_10pct': np.mean(errors < 10.0),
                    'median_error': np.median(errors),
                    'worst_case': np.max(errors),
                    'best_case': np.min(errors),
                    'bias': np.mean(errors) - 5.0,  # Assuming 5% is target
                    'difficulty_rank': np.mean(errors)
                }
        
        # Rank parameters by difficulty
        if param_results:
            difficulties = [(alg, data['mean_error']) for alg, data in param_results.items()]
            difficulties.sort(key=lambda x: x[1])
            
            results[param_name] = {
                'algorithm_performance': param_results,
                'difficulty_ranking': [alg for alg, _ in difficulties],
                'overall_difficulty': np.mean([data['mean_error'] for data in param_results.values()])
            }
    
    # Parameter difficulty analysis across all algorithms
    param_difficulty = {}
    for param, param_name in zip(param_cols, param_names):
        all_errors = df[param].dropna().values
        if len(all_errors) > 0:
            param_difficulty[param_name] = {
                'mean_error': np.mean(all_errors),
                'success_rate': np.mean(all_errors < 5.0),
                'rank': 0  # Will be filled below
            }
    
    # Rank parameters by overall difficulty
    sorted_params = sorted(param_difficulty.items(), key=lambda x: x[1]['mean_error'], reverse=True)
    for rank, (param_name, data) in enumerate(sorted_params, 1):
        param_difficulty[param_name]['rank'] = rank
    
    results['overall_parameter_difficulty'] = param_difficulty
    
    return results

def generate_summary_report(df, significance_results, robustness_results, multivariate_results, 
                          efficiency_results, convergence_results, parameter_results):
    algorithms = df['Algorithm'].unique()
    
    report = {
        'dataset_summary': {
            'total_runs': len(df),
            'algorithms': list(algorithms),
            'runs_per_algorithm': {alg: len(df[df['Algorithm'] == alg]) for alg in algorithms}
        },
        
        'performance_ranking': {},
        'best_algorithm_by_metric': {},
        'algorithm_strengths': {}
    }
    
    # Performance ranking by key metrics
    key_metrics = ['Average_Error_Pct', 'Time_s', 'R_Squared', 'Final_Cost']
    
    for metric in key_metrics:
        if metric in robustness_results:
            metric_means = {alg: data['mean'] for alg, data in robustness_results[metric].items()}
            
            if 'Error' in metric or 'Cost' in metric or 'Time' in metric:
                best_alg = min(metric_means, key=metric_means.get)
                ranked = sorted(metric_means.items(), key=lambda x: x[1])
            else:
                best_alg = max(metric_means, key=metric_means.get)
                ranked = sorted(metric_means.items(), key=lambda x: x[1], reverse=True)
            
            report['best_algorithm_by_metric'][metric] = best_alg
            report['performance_ranking'][metric] = ranked
    
    # Algorithm strengths
    for alg in algorithms:
        strengths = []
        
        # Check if best in any metric
        best_metrics = [metric for metric, best_alg in report['best_algorithm_by_metric'].items() if best_alg == alg]
        if best_metrics:
            strengths.extend([f"Best {metric}" for metric in best_metrics])
        
        # Check convergence
        if alg in convergence_results:
            success_rate = convergence_results[alg]['overall_success_rate']
            if success_rate > 0.8:
                strengths.append("High convergence rate")
            elif success_rate > 0.6:
                strengths.append("Good convergence rate")
        
        # Check robustness
        if 'Average_Error_Pct' in robustness_results:
            cv = robustness_results['Average_Error_Pct'][alg]['cv']
            if cv < 50:
                strengths.append("High consistency")
            elif cv < 100:
                strengths.append("Good consistency")
        
        report['algorithm_strengths'][alg] = strengths
    
    return report

def run_complete_analysis(filepath, output_file=None):
    df = load_clean_data(filepath)
    
    # Define metrics for analysis
    performance_metrics = ['Average_Error_Pct', 'Final_Cost', 'Time_s', 'R_Squared', 'RMSE_Total', 'Max_Param_Error']
    error_metrics = ['MSE_Current', 'MSE_Torque', 'MSE_RPM', 'MAE_Current', 'MAE_Torque', 'MAE_RPM']
    
    all_metrics = performance_metrics + error_metrics
    
    print("Running statistical analysis...")
    
    # Run all analyses
    significance_results = statistical_significance_tests(df, all_metrics)
    robustness_results = robustness_analysis(df, all_metrics)
    multivariate_results = multivariate_analysis(df, performance_metrics)
    efficiency_results = efficiency_analysis(df)
    convergence_results = convergence_analysis(df)
    parameter_results = parameter_specific_analysis(df)
    
    # Generate summary
    summary_report = generate_summary_report(df, significance_results, robustness_results, 
                                           multivariate_results, efficiency_results, 
                                           convergence_results, parameter_results)
    
    results = {
        'summary': summary_report,
        'statistical_significance': significance_results,
        'robustness_analysis': robustness_results,
        'multivariate_analysis': multivariate_results,
        'efficiency_analysis': efficiency_results,
        'convergence_analysis': convergence_results,
        'parameter_analysis': parameter_results
    }
    
    if output_file:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {output_file}")
    
    print("Analysis completed.")
    return results

if __name__ == "__main__":
    results = run_complete_analysis("parameter_identification_results.csv", "complete_analysis_results.pkl")