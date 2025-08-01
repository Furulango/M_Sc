#!/usr/bin/env python3

import sys
import os
from statistical_analysis import run_complete_analysis
from report_generator import load_and_generate_report, StatisticalReportGenerator
import pickle

def main():
    csv_file = "D:\GitHub\M_Sc\TESIS\Congreso\CIMCIA\Final\parameter_identification_results.csv"
    results_file = "complete_analysis_results.pkl"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    print("Starting comprehensive statistical analysis...")
    print("="*60)
    
    # Run complete analysis
    try:
        results = run_complete_analysis(csv_file, results_file)
        print("✓ Statistical analysis completed successfully")
    except Exception as e:
        print(f"✗ Error in statistical analysis: {e}")
        return
    
    # Generate reports
    try:
        generator = load_and_generate_report(results_file)
        print("✓ Reports generated successfully")
    except Exception as e:
        print(f"✗ Error generating reports: {e}")
        return
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files generated:")
    print("  - complete_analysis_results.pkl")
    print("  - analysis_report.txt")
    print("  - algorithm_comparison.png")
    print("  - robustness_analysis.png") 
    print("  - parameter_analysis.png")
    print("  - tables/ (LaTeX tables)")

def print_quick_summary():
    """Print quick summary without full analysis"""
    try:
        with open("complete_analysis_results.pkl", 'rb') as f:
            results = pickle.load(f)
        
        print("\nQUICK SUMMARY:")
        print("-" * 40)
        
        summary = results['summary']
        
        print(f"Dataset: {summary['dataset_summary']['total_runs']} total runs")
        print(f"Algorithms: {', '.join(summary['dataset_summary']['algorithms'])}")
        
        print("\nBest algorithm by metric:")
        for metric, best_alg in summary['best_algorithm_by_metric'].items():
            print(f"  {metric}: {best_alg}")
        
        print("\nAlgorithm rankings (Average Error %):")
        if 'Average_Error_Pct' in summary['performance_ranking']:
            for i, (alg, error) in enumerate(summary['performance_ranking']['Average_Error_Pct'], 1):
                print(f"  {i}. {alg}: {error:.2f}%")
        
    except FileNotFoundError:
        print("No analysis results found. Run main analysis first.")

def run_specific_analysis(analysis_type):
    """Run specific analysis module"""
    csv_file = "parameter_identification_results.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    from statistical_analysis import (load_clean_data, statistical_significance_tests, 
                                    robustness_analysis, multivariate_analysis,
                                    efficiency_analysis, convergence_analysis, 
                                    parameter_specific_analysis)
    
    df = load_clean_data(csv_file)
    metrics = ['Average_Error_Pct', 'Final_Cost', 'Time_s', 'R_Squared']
    
    if analysis_type == "significance":
        print("Running statistical significance tests...")
        results = statistical_significance_tests(df, metrics)
        
        print("\nSignificant differences found in:")
        for metric, test_result in results.items():
            if test_result['omnibus_test']['significant']:
                print(f"  - {metric} (p={test_result['omnibus_test']['p_value']:.4f})")
    
    elif analysis_type == "robustness":
        print("Running robustness analysis...")
        results = robustness_analysis(df, metrics)
        
        print("\nCoefficient of Variation by algorithm:")
        for metric in metrics:
            if metric in results:
                print(f"\n{metric}:")
                for alg, data in results[metric].items():
                    print(f"  {alg}: {data['cv']:.2f}%")
    
    elif analysis_type == "efficiency":
        print("Running efficiency analysis...")
        results = efficiency_analysis(df)
        
        print("\nEfficiency ranking:")
        eff_ranking = sorted(results.items(), key=lambda x: x[1]['efficiency_index'])
        for i, (alg, data) in enumerate(eff_ranking, 1):
            print(f"  {i}. {alg}: {data['efficiency_index']:.4f}")
    
    elif analysis_type == "convergence":
        print("Running convergence analysis...")
        results = convergence_analysis(df)
        
        print("\nConvergence success rates:")
        for alg, data in results.items():
            print(f"  {alg}: {data['overall_success_rate']*100:.1f}%")
    
    elif analysis_type == "parameters":
        print("Running parameter-specific analysis...")
        results = parameter_specific_analysis(df)
        
        print("\nParameter difficulty ranking:")
        if 'overall_parameter_difficulty' in results:
            sorted_params = sorted(results['overall_parameter_difficulty'].items(), 
                                 key=lambda x: x[1]['rank'])
            for param, data in sorted_params:
                print(f"  {data['rank']}. {param}: {data['mean_error']:.2f}% error")
    
    else:
        print(f"Unknown analysis type: {analysis_type}")
        print("Available types: significance, robustness, efficiency, convergence, parameters")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "summary":
            print_quick_summary()
        elif command in ["significance", "robustness", "efficiency", "convergence", "parameters"]:
            run_specific_analysis(command)
        elif command == "help":
            print("Usage:")
            print("  python main_analysis.py              # Run complete analysis")
            print("  python main_analysis.py summary      # Show quick summary")
            print("  python main_analysis.py significance # Significance tests only")
            print("  python main_analysis.py robustness   # Robustness analysis only")
            print("  python main_analysis.py efficiency   # Efficiency analysis only")
            print("  python main_analysis.py convergence  # Convergence analysis only")
            print("  python main_analysis.py parameters   # Parameter analysis only")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for available commands")
    else:
        main()