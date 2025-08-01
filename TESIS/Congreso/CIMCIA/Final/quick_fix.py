#!/usr/bin/env python3

import pandas as pd
import numpy as np

def quick_data_fix():
    """Quick fix for the most common data issues"""
    
    print("Quick Data Fix for Statistical Analysis")
    print("="*50)
    
    # Load and examine data
    try:
        df = pd.read_csv("D:\GitHub\M_Sc\TESIS\Congreso\CIMCIA\Final\parameter_identification_results.csv")
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Remove comment lines
    print("Removing comment lines...")
    original_size = len(df)
    df_clean = df[~df['Algorithm'].str.contains('#|Dataset', na=False, case=False)]
    df_clean = df_clean.dropna(subset=['Algorithm'])
    print(f"  {original_size} → {len(df_clean)} rows")
    
    # Check algorithms
    algorithms = df_clean['Algorithm'].unique()
    print(f"Algorithms: {list(algorithms)}")
    
    # Fix the most problematic columns
    print("Fixing data types...")
    
    # Boolean columns
    bool_cols = ['Converged_Successfully', 'Criterion_Error10pct', 'Criterion_Cost001', 
                'Criterion_5Params5pct', 'Criterion_R2_95pct']
    
    for col in bool_cols:
        if col in df_clean.columns:
            try:
                # Convert string boolean to actual boolean
                df_clean[col] = df_clean[col].astype(str).str.lower()
                df_clean[col] = df_clean[col].map({'true': True, 'false': False, 'nan': False})
                df_clean[col] = df_clean[col].fillna(False)
                print(f"  ✓ Fixed {col}")
            except Exception as e:
                print(f"  ⚠ Issue with {col}: {e}")
    
    # Numeric columns - force conversion
    numeric_cols = ['Time_s', 'Final_Cost', 'Average_Error_Pct', 'R_Squared', 
                   'MSE_Current', 'MSE_Torque', 'MSE_RPM', 'RMSE_Total',
                   'MAE_Current', 'MAE_Torque', 'MAE_RPM']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            try:
                # Force numeric conversion
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Replace infinities with NaN
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                null_count = df_clean[col].isnull().sum()
                print(f"  ✓ Fixed {col} ({null_count} NaN values)")
            except Exception as e:
                print(f"  ⚠ Issue with {col}: {e}")
    
    # Save cleaned data
    df_clean.to_csv("clean_parameter_results.csv", index=False)
    print(f"✓ Saved clean data to 'clean_parameter_results.csv'")
    
    # Quick test of basic statistics
    print("\nQuick validation:")
    for alg in algorithms:
        alg_data = df_clean[df_clean['Algorithm'] == alg]
        n_runs = len(alg_data)
        
        if 'Average_Error_Pct' in alg_data.columns:
            avg_error = alg_data['Average_Error_Pct'].mean()
            print(f"  {alg}: {n_runs} runs, avg error = {avg_error:.2f}%")
        else:
            print(f"  {alg}: {n_runs} runs")
    
    return df_clean

def test_basic_analysis():
    """Test basic statistical functions"""
    
    print("\nTesting basic analysis functions...")
    
    try:
        df = pd.read_csv("clean_parameter_results.csv")
        algorithms = df['Algorithm'].unique()
        
        print("Basic statistics test:")
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            
            # Test basic statistics
            if 'Average_Error_Pct' in alg_data.columns:
                error_values = alg_data['Average_Error_Pct'].dropna()
                if len(error_values) > 0:
                    mean_val = float(error_values.mean())
                    std_val = float(error_values.std())
                    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                    
                    print(f"  {alg}: mean={mean_val:.2f}, std={std_val:.2f}, CV={cv:.1f}%")
        
        print("✓ Basic analysis functions working")
        return True
        
    except Exception as e:
        print(f"✗ Basic analysis failed: {e}")
        return False

def run_simple_comparison():
    """Run a simple comparison without complex statistics"""
    
    print("\nSimple Algorithm Comparison:")
    print("-" * 40)
    
    try:
        df = pd.read_csv("clean_parameter_results.csv")
        algorithms = df['Algorithm'].unique()
        
        results = {}
        
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            
            results[alg] = {
                'n_runs': len(alg_data),
                'avg_error': alg_data['Average_Error_Pct'].mean(),
                'std_error': alg_data['Average_Error_Pct'].std(),
                'avg_time': alg_data['Time_s'].mean() if 'Time_s' in alg_data.columns else 0,
                'success_rate': alg_data['Converged_Successfully'].mean() * 100 if 'Converged_Successfully' in alg_data.columns else 0
            }
        
        # Sort by average error (lower is better)
        sorted_algs = sorted(results.items(), key=lambda x: x[1]['avg_error'])
        
        print("Ranking by Average Error (lower is better):")
        for rank, (alg, data) in enumerate(sorted_algs, 1):
            print(f"{rank}. {alg}:")
            print(f"   Error: {data['avg_error']:.2f}% ± {data['std_error']:.2f}%")
            print(f"   Time: {data['avg_time']:.1f}s")
            print(f"   Success: {data['success_rate']:.1f}%")
            print(f"   Runs: {data['n_runs']}")
            print()
        
        return results
        
    except Exception as e:
        print(f"✗ Simple comparison failed: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Fix data
    df_clean = quick_data_fix()
    
    # Step 2: Test basic functions
    if test_basic_analysis():
        # Step 3: Run simple comparison
        results = run_simple_comparison()
        
        if results:
            print("✓ Data is ready for advanced statistical analysis")
            print("You can now run: python main_analysis.py")
        else:
            print("⚠ Some issues remain, but basic analysis works")
    else:
        print("✗ Data still has issues that need manual inspection")