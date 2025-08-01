# Metrics Documentation - Induction Motor Parameter Identification

## General Information

This dataset contains the results of bio-inspired optimization algorithms applied to parameter identification of a three-phase induction motor. Each row represents an independent execution of an algorithm.

### Real Motor Parameters
- **rs**: 2.45 Ω (Stator resistance)
- **rr**: 1.83 Ω (Rotor resistance)  
- **Lls**: 0.008 H (Stator leakage inductance)
- **Llr**: 0.008 H (Rotor leakage inductance)
- **Lm**: 0.203 H (Magnetizing inductance)
- **J**: 0.02 kg⋅m² (Moment of inertia)
- **B**: 0.001 N⋅m⋅s/rad (Viscous friction coefficient)

### Experimental Conditions
- **Noise**: 3% added to experimental signals
- **Search limits**: ±50% of real values
- **Simulation duration**: 2.0 seconds
- **Data points**: 500 points per signal

---

## Metrics Description

### BASIC IDENTIFICATION

| Metric | Description | Type | Possible Values |
|---------|-------------|------|------------------|
| `Algorithm` | Name of algorithm used | String | PSO, PSO-SQP, BFO |
| `Run` | Execution number | Integer | 1-30 |
| `Seed` | Random seed used | Integer | Random 1-999999 |
| `Configuration` | Algorithm configuration | String | PSO_30p_50i, BFO_20b_15q_3r |

### PERFORMANCE METRICS

| Metric | Description | Unit | Interpretation |
|---------|-------------|------|----------------|
| `Time_s` | Execution time | Seconds | Lower = more efficient |
| `Final_Cost` | Final objective function value | Dimensionless | Lower = better fit |
| `Best_Cost_Iter` | Best cost achieved during execution | Dimensionless | Lower = better convergence |
| `Num_FO_Evaluations` | Number of objective function evaluations | Integer | Higher = more computationally expensive |
| `Convergence_Iteration` | Iteration where it converged | Integer | -1 = did not converge |

### SUCCESSFUL CONVERGENCE CRITERIA

| Metric | Description | Type | Criterion |
|---------|-------------|------|----------|
| `Converged_Successfully` | If algorithm converged successfully | Boolean | True if meets ≥2 criteria |
| `Criterion_Error10pct` | Average error < 10% | Boolean | True/False |
| `Criterion_Cost001` | Final cost < 0.01 | Boolean | True/False |
| `Criterion_5Params5pct` | ≥5 parameters with error < 5% | Boolean | True/False |
| `Criterion_R2_95pct` | R² > 0.95 | Boolean | True/False |
| `Criteria_Met` | Number of criteria met | Integer | 0-4 |

**Note**: An algorithm "converges successfully" if it meets **AT LEAST 2** of the 4 previous criteria. This means the result is **True** when at least 2 of the 4 criteria are true.

**Criteria Explanation**:
- **Criterion 1**: Average error of all parameters less than 10%
- **Criterion 2**: Final objective function less than 0.01 
- **Criterion 3**: At least 5 of the 7 parameters have individual error less than 5%
- **Criterion 4**: R² coefficient greater than 0.95 (good fit)

### MSE ERRORS (Mean Square Error)

| Metric | Description | Unit | Typical Range |
|---------|-------------|------|--------------|
| `MSE_Current` | MSE of current magnitude | A² | 10⁻⁶ - 10⁻² |
| `MSE_Torque` | MSE of electromagnetic torque | (N⋅m)² | 10⁻⁶ - 10⁻² |
| `MSE_RPM` | MSE of speed | RPM² | 10⁻⁴ - 10² |
| `RMSE_Total` | Combined weighted RMSE | Dimensionless | 10⁻³ - 10⁻¹ |

**RMSE_Total Formula**: √(1.0×MSE_current + 0.5×MSE_torque + 0.3×MSE_rpm)

### MAE ERRORS (Maximum Absolute Error)

| Metric | Description | Unit | Interpretation |
|---------|-------------|------|----------------|
| `MAE_Current` | Maximum absolute error in current | A | Worst case in current |
| `MAE_Torque` | Maximum absolute error in torque | N⋅m | Worst case in torque |
| `MAE_RPM` | Maximum absolute error in RPM | RPM | Worst case in speed |

### IDENTIFIED PARAMETERS

| Metric | Description | Unit | Real Value |
|---------|-------------|------|------------|
| `rs_identified` | Identified stator resistance | Ω | 2.45 |
| `rr_identified` | Identified rotor resistance | Ω | 1.83 |
| `Lls_identified` | Identified stator leakage inductance | H | 0.008 |
| `Llr_identified` | Identified rotor leakage inductance | H | 0.008 |
| `Lm_identified` | Identified magnetizing inductance | H | 0.203 |
| `J_identified` | Identified moment of inertia | kg⋅m² | 0.02 |
| `B_identified` | Identified viscous friction coefficient | N⋅m⋅s/rad | 0.001 |

### ERRORS PER PARAMETER

| Metric | Description | Unit | Interpretation |
|---------|-------------|------|----------------|
| `Error_pct_rs` | Percentage error in rs | % | \|rs_real - rs_id\|/rs_real × 100 |
| `Error_pct_rr` | Percentage error in rr | % | \|rr_real - rr_id\|/rr_real × 100 |
| `Error_pct_Lls` | Percentage error in Lls | % | \|Lls_real - Lls_id\|/Lls_real × 100 |
| `Error_pct_Llr` | Percentage error in Llr | % | \|Llr_real - Llr_id\|/Llr_real × 100 |
| `Error_pct_Lm` | Percentage error in Lm | % | \|Lm_real - Lm_id\|/Lm_real × 100 |
| `Error_pct_J` | Percentage error in J | % | \|J_real - J_id\|/J_real × 100 |
| `Error_pct_B` | Percentage error in B | % | \|B_real - B_id\|/B_real × 100 |

### SUMMARY METRICS

| Metric | Description | Unit | Interpretation |
|---------|-------------|------|----------------|
| `Average_Error_Pct` | Average of percentage errors | % | Lower = better global identification |
| `Average_Absolute_Error` | Average of absolute errors | Mixed | Average of \|param_real - param_id\| |
| `Euclidean_Distance` | Euclidean distance to real vector | Dimensionless | ‖params_real - params_id‖ |
| `Max_Param_Error` | Worst error among all parameters | % | max(Error_pct_rs, ..., Error_pct_B) |
| `Good_Params_5pct` | Parameters with error < 5% | Integer | 0-7, more is better |

### ADVANCED STATISTICAL METRICS

| Metric | Description | Unit | Range | Interpretation |
|---------|-------------|--------|-------|----------------|
| `R_Squared` | Coefficient of determination | Dimensionless | 0-1 | Fit quality (1=perfect) |
| `MAPE_Average` | Mean absolute percentage error | % | 0-∞ | Average of relative errors |
| `Theil_Coefficient` | Theil's inequality coefficient | Dimensionless | 0-1 | Prediction precision (0=perfect) |

**Formulas**:
- **R²**: 1 - (SS_res/SS_tot), where SS_res = Σ(y_real - y_pred)², SS_tot = Σ(y_real - ȳ)²
- **MAPE**: (1/n) × Σ\|y_real - y_pred\|/y_real × 100
- **Theil**: √(MSE) / (√(Σy_real²/n) + √(Σy_pred²/n))

---

## Interpretation Guide

### Excellent Identification (Publishable)
- `Average_Error_Pct` < 5%
- `R_Squared` > 0.98
- `Good_Params_5pct` ≥ 6
- `Converged_Successfully` = True

### Good Identification (Acceptable)
- `Average_Error_Pct` < 10%
- `R_Squared` > 0.95
- `Good_Params_5pct` ≥ 4
- `Converged_Successfully` = True

### Regular Identification (Improvable)
- `Average_Error_Pct` < 20%
- `R_Squared` > 0.90
- `Good_Params_5pct` ≥ 2

### Poor Identification (Reject)
- `Average_Error_Pct` > 20%
- `R_Squared` < 0.90
- `Good_Params_5pct` < 2
- `Converged_Successfully` = False

---

## Statistical Analysis Suggestions

### Algorithm Comparison
1. **Precision**: Compare `Average_Error_Pct` by algorithm
2. **Consistency**: Analyze std of `Final_Cost` by algorithm  
3. **Efficiency**: Compare `Time_s` and `Num_FO_Evaluations`
4. **Robustness**: Analyze `Converged_Successfully` (success rate)

### Recommended Statistical Tests
- **ANOVA**: To compare means between algorithms
- **Kruskal-Wallis**: If data is not normal
- **Wilcoxon**: For paired comparisons
- **Chi-square**: For `Converged_Successfully`

### Suggested Visualizations
- **Boxplots**: `Average_Error_Pct` by algorithm
- **Scatter**: `Time_s` vs `Average_Error_Pct`
- **Histograms**: Distribution of `R_Squared`
- **Heatmap**: Correlation between metrics

### Example Code for Analysis

#### Python/Pandas:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('parameter_identification_results.csv')

# Summary by algorithm
summary = df.groupby('Algorithm').agg({
    'Average_Error_Pct': ['mean', 'std', 'min', 'max'],
    'Time_s': ['mean', 'std'],
    'Converged_Successfully': 'mean',
    'R_Squared': ['mean', 'std']
}).round(3)

print(summary)

# Comparative boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Algorithm', y='Average_Error_Pct')
plt.title('Comparison of Average Error by Algorithm')
plt.ylabel('Average Error (%)')
plt.show()

# Convergence rate
convergence_rate = df.groupby('Algorithm')['Converged_Successfully'].mean()
print(f"\nSuccessful Convergence Rate:")
print(convergence_rate * 100)
```

#### R:
```r
# Load data
df <- read.csv('parameter_identification_results.csv')

# Statistical summary
summary(df)

# ANOVA for Average_Error_Pct
anova_result <- aov(Average_Error_Pct ~ Algorithm, data=df)
summary(anova_result)

# Post-hoc test
TukeyHSD(anova_result)

# Boxplot
boxplot(Average_Error_Pct ~ Algorithm, data=df,
        main="Comparison of Average Error by Algorithm",
        ylab="Average Error (%)")

# Chi-square test for convergence
contingency_table <- table(df$Algorithm, df$Converged_Successfully)
chisq.test(contingency_table)
```

---

## Additional Notes

### Reproducibility
- **Unique random seeds**: Each run uses a truly random seed (1-999999)
- **Automatic verification**: System automatically verifies that seeds are not repeated
- **Statistical independence**: Guarantees real independence between executions
- **Individual reproducibility**: With same seed, same exact result is obtained
- **Permanent record**: All seeds are stored in CSV for traceability

### Data Handling
- **Immediate saving**: Each execution is saved immediately upon completion
- **Interruption tolerance**: If process is interrupted, previous data is not lost
- **Missing Values**: -1 in `Convergence_Iteration` indicates algorithm did not converge
- **Outliers**: Extreme values may indicate algorithm failure or inadequate configuration
- **Scales**: Some metrics have very different scales, consider normalization for multivariate analysis
- **Duplicate verification**: Automatic system prevents executions with repeated seeds

### Publication Considerations
- **Sample size**: 30 executions is appropriate for basic statistical analysis
- **Significance**: Use α = 0.05 for hypothesis testing
- **Practical effect**: Not only statistical significance, but also practical relevance
- **Confidence intervals**: Report 95% CI along with means

---

**Version**: 1.0  
**Date**: 2024  
**File format**: CSV with headers  
**Encoding**: UTF-8  
**Separator**: Comma (,)