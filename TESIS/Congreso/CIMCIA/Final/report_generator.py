import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import pickle

class StatisticalReportGenerator:
    def __init__(self, results_dict):
        self.results = results_dict
        self.algorithms = list(self.results['summary']['dataset_summary']['algorithms'])
        
    def create_significance_table(self):
        significance_data = []
        
        for metric, tests in self.results['statistical_significance'].items():
            omnibus = tests['omnibus_test']
            significance_data.append({
                'Metric': metric,
                'Test': omnibus['test'],
                'Statistic': f"{omnibus['statistic']:.4f}",
                'P-value': f"{omnibus['p_value']:.4f}",
                'Significant': 'Yes' if omnibus['significant'] else 'No'
            })
        
        return pd.DataFrame(significance_data)
    
    def create_robustness_table(self):
        robustness_data = []
        
        for metric in ['Average_Error_Pct', 'Time_s', 'Final_Cost', 'R_Squared']:
            if metric in self.results['robustness_analysis']:
                for alg in self.algorithms:
                    if alg in self.results['robustness_analysis'][metric]:
                        data = self.results['robustness_analysis'][metric][alg]
                        robustness_data.append({
                            'Algorithm': alg,
                            'Metric': metric,
                            'Mean': f"{data['mean']:.4f}",
                            'Std': f"{data['std']:.4f}",
                            'CV(%)': f"{data['cv']:.2f}",
                            'Median': f"{data['median']:.4f}",
                            'IQR': f"{data['iqr']:.4f}",
                            'Robustness_Index': f"{data['robustness_index']:.4f}"
                        })
        
        return pd.DataFrame(robustness_data)
    
    def create_pairwise_comparison_table(self):
        pairwise_data = []
        
        key_metrics = ['Average_Error_Pct', 'Time_s', 'Final_Cost']
        
        for metric in key_metrics:
            if metric in self.results['statistical_significance']:
                comparisons = self.results['statistical_significance'][metric]['pairwise_comparisons']
                for comparison, data in comparisons.items():
                    pairwise_data.append({
                        'Metric': metric,
                        'Comparison': comparison.replace('_vs_', ' vs '),
                        'P-value': f"{data['p_value']:.4f}",
                        'Effect_Size': f"{data['effect_size']:.4f}",
                        'Significant': 'Yes' if data['significant'] else 'No'
                    })
        
        return pd.DataFrame(pairwise_data)
    
    def create_efficiency_table(self):
        efficiency_data = []
        
        for alg, data in self.results['efficiency_analysis'].items():
            efficiency_data.append({
                'Algorithm': alg,
                'Mean_Error(%)': f"{data['mean_error']:.2f}",
                'Mean_Time(s)': f"{data['mean_time']:.2f}",
                'Efficiency_Index': f"{data['efficiency_index']:.4f}",
                'Error_Time_Ratio': f"{data['error_time_ratio']:.4f}",
                'Normalized_Performance': f"{data['normalized_performance']:.4f}"
            })
        
        return pd.DataFrame(efficiency_data)
    
    def create_convergence_table(self):
        convergence_data = []
        
        for alg, data in self.results['convergence_analysis'].items():
            convergence_data.append({
                'Algorithm': alg,
                'Overall_Success(%)': f"{data['overall_success_rate']*100:.1f}",
                'Mean_Criteria_Met': f"{data['mean_criteria_met']:.2f}",
                'Prob_All_Criteria(%)': f"{data['probability_all_criteria']*100:.1f}",
                'Prob_Most_Criteria(%)': f"{data['probability_most_criteria']*100:.1f}",
                'Prob_Some_Criteria(%)': f"{data['probability_some_criteria']*100:.1f}"
            })
        
        return pd.DataFrame(convergence_data)
    
    def create_parameter_difficulty_table(self):
        param_data = []
        
        overall_difficulty = self.results['parameter_analysis']['overall_parameter_difficulty']
        
        for param, data in overall_difficulty.items():
            param_data.append({
                'Parameter': param,
                'Mean_Error(%)': f"{data['mean_error']:.2f}",
                'Success_Rate_5%(%)': f"{data['success_rate']*100:.1f}",
                'Difficulty_Rank': data['rank']
            })
        
        return pd.DataFrame(param_data).sort_values('Difficulty_Rank')
    
    def create_composite_ranking_table(self):
        ranking_data = self.results['multivariate_analysis']['composite_ranking']
        
        ranking_table = []
        for i, (alg, data) in enumerate(ranking_data.iterrows(), 1):
            ranking_table.append({
                'Rank': i,
                'Algorithm': alg,
                'Composite_Score': f"{data['mean']:.4f}",
                'Score_Std': f"{data['std']:.4f}"
            })
        
        return pd.DataFrame(ranking_table)
    
    def plot_algorithm_comparison(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        # Error comparison
        error_data = []
        for alg in self.algorithms:
            if alg in self.results['robustness_analysis']['Average_Error_Pct']:
                error_data.append(self.results['robustness_analysis']['Average_Error_Pct'][alg]['mean'])
        
        axes[0,0].bar(self.algorithms, error_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0,0].set_title('Average Error Percentage')
        axes[0,0].set_ylabel('Error (%)')
        
        # Time comparison
        time_data = []
        for alg in self.algorithms:
            if alg in self.results['robustness_analysis']['Time_s']:
                time_data.append(self.results['robustness_analysis']['Time_s'][alg]['mean'])
        
        axes[0,1].bar(self.algorithms, time_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0,1].set_title('Average Execution Time')
        axes[0,1].set_ylabel('Time (s)')
        
        # Convergence rates
        conv_data = []
        for alg in self.algorithms:
            conv_data.append(self.results['convergence_analysis'][alg]['overall_success_rate']*100)
        
        axes[1,0].bar(self.algorithms, conv_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1,0].set_title('Convergence Success Rate')
        axes[1,0].set_ylabel('Success Rate (%)')
        
        # Efficiency scatter
        for i, alg in enumerate(self.algorithms):
            eff_data = self.results['efficiency_analysis'][alg]
            axes[1,1].scatter(eff_data['mean_error'], eff_data['mean_time'], 
                            s=100, label=alg, alpha=0.7)
        
        axes[1,1].set_xlabel('Mean Error (%)')
        axes[1,1].set_ylabel('Mean Time (s)')
        axes[1,1].set_title('Error vs Time Trade-off')
        axes[1,1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_robustness_analysis(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Robustness Analysis', fontsize=16)
        
        metrics = ['Average_Error_Pct', 'Time_s', 'Final_Cost', 'R_Squared']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            if metric in self.results['robustness_analysis']:
                cvs = []
                for alg in self.algorithms:
                    if alg in self.results['robustness_analysis'][metric]:
                        cvs.append(self.results['robustness_analysis'][metric][alg]['cv'])
                    else:
                        cvs.append(0)
                
                bars = ax.bar(self.algorithms, cvs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                ax.set_title(f'Coefficient of Variation - {metric}')
                ax.set_ylabel('CV (%)')
                
                # Add value labels on bars
                for bar, cv in zip(bars, cvs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(cvs),
                           f'{cv:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_analysis(self, save_path=None):
        param_names = list(self.results['parameter_analysis']['overall_parameter_difficulty'].keys())
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Parameter Identification Analysis', fontsize=16)
        
        # Parameter difficulty
        difficulties = [self.results['parameter_analysis']['overall_parameter_difficulty'][param]['mean_error'] 
                       for param in param_names]
        
        bars = axes[0].bar(param_names, difficulties, color='lightcoral')
        axes[0].set_title('Parameter Identification Difficulty')
        axes[0].set_ylabel('Mean Error (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Algorithm performance by parameter
        param_performance = {}
        for param in param_names:
            if param in self.results['parameter_analysis']:
                for alg in self.algorithms:
                    if alg not in param_performance:
                        param_performance[alg] = []
                    if alg in self.results['parameter_analysis'][param]['algorithm_performance']:
                        param_performance[alg].append(
                            self.results['parameter_analysis'][param]['algorithm_performance'][alg]['mean_error']
                        )
                    else:
                        param_performance[alg].append(0)
        
        x = np.arange(len(param_names))
        width = 0.25
        
        for i, alg in enumerate(self.algorithms):
            if alg in param_performance:
                axes[1].bar(x + i*width, param_performance[alg], width, 
                           label=alg, alpha=0.8)
        
        axes[1].set_title('Algorithm Performance by Parameter')
        axes[1].set_ylabel('Mean Error (%)')
        axes[1].set_xlabel('Parameters')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(param_names)
        axes[1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_latex_tables(self, output_dir="tables/"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        tables = {
            'significance_tests': self.create_significance_table(),
            'robustness_analysis': self.create_robustness_table(),
            'pairwise_comparisons': self.create_pairwise_comparison_table(),
            'efficiency_analysis': self.create_efficiency_table(),
            'convergence_analysis': self.create_convergence_table(),
            'parameter_difficulty': self.create_parameter_difficulty_table(),
            'composite_ranking': self.create_composite_ranking_table()
        }
        
        for name, df in tables.items():
            latex_str = df.to_latex(index=False, escape=False, 
                                  caption=f"{name.replace('_', ' ').title()} Results",
                                  label=f"tab:{name}")
            
            with open(f"{output_dir}{name}.tex", 'w') as f:
                f.write(latex_str)
        
        print(f"LaTeX tables saved to {output_dir}")
    
    def generate_full_report(self, output_file="analysis_report.txt"):
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL ANALYSIS REPORT - ALGORITHM COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY:\n")
            f.write("-"*40 + "\n")
            summary = self.results['summary']['dataset_summary']
            f.write(f"Total runs: {summary['total_runs']}\n")
            f.write(f"Algorithms: {', '.join(summary['algorithms'])}\n")
            for alg, count in summary['runs_per_algorithm'].items():
                f.write(f"  {alg}: {count} runs\n")
            f.write("\n")
            
            # Best algorithms by metric
            f.write("BEST ALGORITHM BY METRIC:\n")
            f.write("-"*40 + "\n")
            for metric, best_alg in self.results['summary']['best_algorithm_by_metric'].items():
                f.write(f"{metric}: {best_alg}\n")
            f.write("\n")
            
            # Algorithm strengths
            f.write("ALGORITHM STRENGTHS:\n")
            f.write("-"*40 + "\n")
            for alg, strengths in self.results['summary']['algorithm_strengths'].items():
                f.write(f"{alg}:\n")
                for strength in strengths:
                    f.write(f"  - {strength}\n")
                f.write("\n")
            
            # Statistical significance summary
            f.write("STATISTICAL SIGNIFICANCE SUMMARY:\n")
            f.write("-"*40 + "\n")
            significant_metrics = []
            for metric, tests in self.results['statistical_significance'].items():
                if tests['omnibus_test']['significant']:
                    significant_metrics.append(metric)
            
            f.write(f"Metrics with significant differences: {len(significant_metrics)}\n")
            for metric in significant_metrics:
                f.write(f"  - {metric}\n")
            
        print(f"Full report saved to {output_file}")

def load_and_generate_report(results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    generator = StatisticalReportGenerator(results)
    
    # Generate all tables
    print("Generating statistical tables...")
    significance_table = generator.create_significance_table()
    robustness_table = generator.create_robustness_table()
    efficiency_table = generator.create_efficiency_table()
    convergence_table = generator.create_convergence_table()
    parameter_table = generator.create_parameter_difficulty_table()
    ranking_table = generator.create_composite_ranking_table()
    
    print("\nSignificance Tests:")
    print(significance_table.to_string(index=False))
    
    print("\nRobustness Analysis:")
    print(robustness_table.to_string(index=False))
    
    print("\nEfficiency Analysis:")
    print(efficiency_table.to_string(index=False))
    
    # Generate plots
    print("\nGenerating plots...")
    generator.plot_algorithm_comparison("algorithm_comparison.png")
    generator.plot_robustness_analysis("robustness_analysis.png")
    generator.plot_parameter_analysis("parameter_analysis.png")
    
    # Generate LaTeX tables
    generator.generate_latex_tables()
    
    # Generate full report
    generator.generate_full_report()
    
    return generator

if __name__ == "__main__":
    generator = load_and_generate_report("complete_analysis_results.pkl")