import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model

# Genetic algorithm (DEAP)
from deap import base, creator, tools, algorithms

# Silenciar advertencias comunes para una salida más limpia
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# 1. DESCARGA Y PREPARACIÓN DE DATOS (Sin cambios)
# -----------------------------------------------------------------------------

def download_data(symbols, start, end):
    """Descarga datos de precios (Close) para los activos."""
    df_full = yf.download(symbols, start=start, end=end, auto_adjust=True)['Close']
    df_full = df_full.ffill().dropna(axis=0, how='any')
    
    print(f"Datos descargados: {len(df_full.columns)} activos.")
    print(f"Periodo: {df_full.index[0].date()} a {df_full.index[-1].date()} ({len(df_full)} dias).")
    
    return df_full

# -----------------------------------------------------------------------------
# 2. ALGORITMO GENÉTICO (VERSIÓN 2 - AJUSTADA)
# -----------------------------------------------------------------------------

def enhanced_genetic_optimizer_v2(expected_returns, cov_matrix, previous_weights=None,
                                  population_size=120, generations=80, cxpb=0.6, mutpb=0.25,
                                  max_position=0.25, risk_free=0.02, min_assets=5):
    """
    Optimizador genético v2 con penalizaciones más estrictas para reducir costos y
    forzar una mayor diversificación.
    
    CAMBIOS CLAVE:
    - Penalización por turnover (rotación) AUMENTADA (10.0 vs 3.0) para reducir costos.
    - Penalización por concentración (Herfindahl) AUMENTADA (2.0 vs 0.8) para mejorar diversificación.
    - Nueva restricción: Se exige un número mínimo de activos en cartera.
    """
    n_assets = len(expected_returns)

    # Recrear clases si no existen para evitar errores en ejecuciones repetidas
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Inicialización inteligente de individuos
    def init_smart_individual():
        if previous_weights is not None:
            base_weights = previous_weights + np.random.normal(0, 0.05, n_assets)
        else:
            inv_vol = 1 / (np.diag(cov_matrix) ** 0.5 + 1e-9)
            base_weights = inv_vol / inv_vol.sum()
        return creator.Individual(np.maximum(base_weights, 0).tolist())

    toolbox.register("individual", init_smart_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_portfolio(individual):
        w = np.array(individual, dtype=float)
        w = np.maximum(w, 0)
        w_sum = w.sum()
        if w_sum < 1e-9: # Evitar división por cero si todos los pesos son cero
            return (-1e10,)
        w = w / w_sum

        # --- NUEVAS RESTRICCIONES Y PENALIZACIONES MÁS DURAS ---
        if np.any(w > max_position):
            return (-1e10,) # Penalización máxima por superar el límite por activo
        
        # NUEVA RESTRICCIÓN: Forzar un número mínimo de activos
        if np.sum(w > 0.01) < min_assets:
            return (-1e10,) # Penalización máxima por baja diversificación

        port_return = np.dot(w, expected_returns)
        
        # Simulación para el Sortino Ratio
        returns_sim = np.random.multivariate_normal(expected_returns, cov_matrix, 500) # Reducido para velocidad
        portfolio_returns_sim = returns_sim @ w
        
        downside_returns = portfolio_returns_sim[portfolio_returns_sim < risk_free / 252] # Usar risk-free diario
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 1 else 1e-9
        sortino_ratio = (port_return - risk_free) / (downside_deviation * np.sqrt(252) + 1e-9)

        # Penalización por concentración (Herfindahl Index)
        herfindahl = np.sum(w**2)
        
        # Penalización por rotación de cartera (Turnover)
        turnover_penalty = 0
        if previous_weights is not None:
            turnover = np.sum(np.abs(w - previous_weights))
            # AJUSTE CLAVE: Penalización mucho más fuerte para reducir costos
            turnover_penalty = turnover * 10.0 
        
        # AJUSTE CLAVE: Función de fitness rebalanceada
        fitness = (sortino_ratio * 1.0 - herfindahl * 2.0 - turnover_penalty)
        return (fitness,)

    toolbox.register("evaluate", evaluate_portfolio)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=float)
    best = np.maximum(best, 0)
    best = best / (best.sum() + 1e-10)
    return best, hof[0].fitness.values[0]


# -----------------------------------------------------------------------------
# 3. ESTRATEGIA Y BACKTESTING (VERSIÓN 2 - LÓGICA CENTRAL MEJORADA)
# -----------------------------------------------------------------------------

def strategy_garch_ga_v2(historical_data, previous_weights=None, rebalance_horizon_days=63, verbose=False):
    """
    ESTRATEGIA V2 MEJORADA:
    1. Calcula retornos esperados usando una Media Móvil Exponencial (EWMA) para mayor estabilidad.
    2. Modela la volatilidad usando GARCH(1,1) pero predice para todo el horizonte de rebalanceo.
    3. Construye la matriz de covarianza híbrida.
    4. Usa el Algoritmo Genético v2 con penalizaciones más estrictas.
    """
    returns = historical_data.pct_change().dropna()
    if len(returns) < 126: # Necesita suficientes datos
        return np.ones(historical_data.shape[1]) / historical_data.shape[1]

    # 1. AJUSTE CLAVE: Calcular retornos esperados con EWMA para dar más peso a datos recientes
    expected_returns_annualized = returns.ewm(span=252, adjust=False).mean().iloc[-1] * 252

    # 2. Predecir volatilidad con GARCH para cada activo, alineando el horizonte
    garch_variances = []
    for asset in returns.columns:
        garch_model = arch_model(returns[asset] * 100, vol='Garch', p=1, q=1, dist='Normal')
        res = garch_model.fit(disp='off', show_warning=False)
        
        # AJUSTE CLAVE: Predecir para todo el horizonte de inversión (ej. 63 días)
        forecast = res.forecast(horizon=rebalance_horizon_days, reindex=False)
        # Usar la media de las varianzas predichas en el horizonte
        pred_variance_period_mean = forecast.variance.iloc[-1].mean()
        
        # Desescalar y anualizar la varianza
        garch_variances.append(pred_variance_period_mean * 252 / (100**2))
    
    # 3. Construir matriz de covarianza híbrida
    corr_matrix = returns.corr()
    garch_volatilities = np.sqrt(garch_variances)
    
    # Crear la matriz de covarianza combinando volatilidades GARCH con correlación histórica
    garch_cov_matrix = pd.DataFrame(np.outer(garch_volatilities, garch_volatilities) * corr_matrix.values,
                                    columns=returns.columns, index=returns.columns)

    if verbose:
        print(" -> Volatilidades GARCH (anualizadas, horizonte-ajustado):", np.round(garch_volatilities, 3))

    # 4. Optimizar con el Algoritmo Genético v2
    try:
        opt_weights, fitness = enhanced_genetic_optimizer_v2(
            expected_returns_annualized.values, 
            garch_cov_matrix.values,
            previous_weights=previous_weights,
            min_assets=max(2, int(len(returns.columns) * 0.4)) # Exigir al menos 40% de los activos
        )
        if verbose:
            print(f" -> GA_v2 finalizado. Fitness: {fitness:.4f}")
            print(" -> Pesos optimizados:", np.round(opt_weights, 3))
    except Exception as e:
        if verbose: print(f" -> ERROR en GA_v2: {e}. Usando inverse-volatility.")
        inv_vol = 1.0 / (garch_volatilities + 1e-9)
        opt_weights = inv_vol / inv_vol.sum()

    return opt_weights

def calculate_transaction_cost(old_weights, new_weights, portfolio_value, cost_rate=0.001):
    turnover = np.sum(np.abs(new_weights - old_weights)) if old_weights is not None else np.sum(new_weights)
    return turnover * cost_rate * portfolio_value

def run_backtest_walk_forward(data, weight_strategy_fn, strategy_name, **kwargs):
    print(f"\n{'='*60}\nEJECUTANDO BACKTEST ({strategy_name})\n{'='*60}")
    
    initial_capital = kwargs.get('initial_capital', 100000)
    training_window = kwargs.get('training_window', 1008)
    rebalance_days = kwargs.get('rebalance_days', 63)
    cost_rate = kwargs.get('transaction_cost_rate', 0.001)

    portfolio_values, dates, weights_history = [], [], []
    current_value = initial_capital
    previous_weights, total_costs = None, 0
    returns = data.pct_change().fillna(0)

    for i in range(training_window, len(data), rebalance_days):
        current_date = data.index[i]
        print(f"\n -> Rebalanceando en {current_date.date()}...")
        
        train_start, train_end = max(0, i - training_window), i
        training_data = data.iloc[train_start:train_end]

        applied_weights = weight_strategy_fn(
            training_data, 
            previous_weights=previous_weights,
            rebalance_horizon_days=rebalance_days,
            verbose=True
        )

        cost = calculate_transaction_cost(previous_weights, applied_weights, current_value, cost_rate)
        current_value -= cost
        total_costs += cost
        print(f" -> Costo de transacción: ${cost:,.2f}")
        
        weights_history.append(applied_weights)
        previous_weights = applied_weights.copy()

        period_end = min(i + rebalance_days, len(data))
        for j in range(i, period_end):
            daily_return = np.dot(applied_weights, returns.iloc[j])
            current_value *= (1 + daily_return)
            portfolio_values.append(current_value)
            dates.append(data.index[j])
            
    print(f"\n -> Total costos de transacción: ${total_costs:,.2f}")
    return pd.Series(portfolio_values, index=dates), np.array(weights_history), total_costs


# -----------------------------------------------------------------------------
# 4. VISUALIZACIÓN Y REPORTES (Sin cambios)
# -----------------------------------------------------------------------------

def calculate_metrics(values, total_costs=0, initial_capital=100000):
    if len(values) < 2: return {}
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    annual_return = (1 + total_return)**(1 / years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / (annual_vol + 1e-9)
    downside_returns = returns[returns < 0.02/252]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - 0.02) / (downside_std + 1e-9)
    drawdown = (values / values.cummax() - 1).min()
    cost_impact = total_costs / values.iloc[-1]
    return {
        'Retorno Anual': f"{annual_return:.2%}", 'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}", 'Sortino Ratio': f"{sortino:.3f}",
        'Max Drawdown': f"{drawdown:.2%}", 'Valor Final': f"${values.iloc[-1]:,.0f}",
        'Costos de Transacción': f"${total_costs:,.2f}", 'Impacto de Costos': f"{cost_impact:.2%}"
    }

def plot_results(results, full_data, training_window, weights_history, symbols, run_dir):
    print("\n" + "="*60 + "\nGENERANDO PANEL DE CONTROL...\n" + "="*60)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Gráfico 1: Evolución de Cartera (Log)
    ax1 = axes[0, 0]
    for name, data in results.items():
      linestyle, color = (':', 'orange') if "Benchmark" in name else ('-', 'tab:blue')
      ax1.plot(data.index, data.values, label=name, linewidth=2.0, linestyle=linestyle, color=color)
    ax1.set_title('1. Evolucion del Valor de Cartera', fontsize=16, weight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, linestyle='--')
    ax1.set_yscale('log')

    # Gráfico 2: Retornos Acumulados
    ax2 = axes[0, 1]
    for name, data in results.items():
        cum_returns = (data / data.iloc[0])
        linestyle, color = (':', 'orange') if "Benchmark" in name else ('-', 'tab:blue')
        ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0, linestyle=linestyle, color=color)
    ax2.set_title('2. Retornos Acumulados', fontsize=16, weight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')

    # Gráfico 3: Distribución Final de Pesos
    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights[final_weights > 0.001], labels=np.array(symbols)[final_weights > 0.001], autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        ax3.set_title('3. Distribucion Final de Pesos', fontsize=16, weight='bold')

    # Gráfico 4: Evolución de Pesos
    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        weights_df = pd.DataFrame(weights_history, columns=symbols)
        weights_df.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis', legend=None)
    ax4.set_title('4. Evolucion de Pesos en Rebalanceos', fontsize=16, weight='bold')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax4.set_xticks(ax4.get_xticks()[::2]) # Mostrar menos etiquetas para legibilidad

    plt.tight_layout(pad=3.0)
    fig.suptitle('Panel de Control - GARCH+GA v2 (Mejorado)', fontsize=22, weight='bold')
    plt.subplots_adjust(top=0.92)
    
    plot_filepath = os.path.join(run_dir, "performance_charts_v2.png")
    fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"Grafico de rendimiento guardado en: {plot_filepath}")
    plt.show()


# -----------------------------------------------------------------------------
# 5. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH']
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    training_window = 1008
    initial_capital = 100000
    rebalance_days = 63
    transaction_cost_rate = 0.001

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = "results"
    run_dir = os.path.join(output_dir, f"run_garch_ga_v2_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("OPTIMIZACION DE CARTERAS: GARCH + ALGORITMO GENÉTICO (VERSIÓN 2 MEJORADA)")
    print(f"Directorio de resultados: {run_dir}")

    try:
        asset_data = download_data(symbols, start_date, end_date)
        
        # Estrategia GARCH + GA v2
        garch_values, garch_weights, garch_costs = run_backtest_walk_forward(
            asset_data, strategy_garch_ga_v2, "GARCH+GA v2",
            training_window=training_window,
            rebalance_days=rebalance_days,
            transaction_cost_rate=transaction_cost_rate,
            initial_capital=initial_capital
        )
        
        # Benchmark Equiponderado (con costos)
        n_assets = asset_data.shape[1]
        benchmark_weights = np.ones(n_assets) / n_assets
        benchmark_values, _, benchmark_costs = run_backtest_walk_forward(
            asset_data, lambda data, **kwargs: benchmark_weights, "Benchmark",
            training_window=training_window,
            rebalance_days=rebalance_days,
            transaction_cost_rate=transaction_cost_rate,
            initial_capital=initial_capital
        )
        
        # Resultados y Reporte
        results = {
            "Estrategia GARCH+GA (Mejorada)": garch_values,
            "Benchmark (Equiponderado)": benchmark_values
        }
        transaction_costs = {
            "Estrategia GARCH+GA (Mejorada)": garch_costs,
            "Benchmark (Equiponderado)": benchmark_costs
        }

        report_lines = ["=" * 60, "METRICAS DE RENDIMIENTO (GARCH + GA v2)", "=" * 60]
        for name, series_data in results.items():
            total_costs = transaction_costs[name]
            metrics = calculate_metrics(series_data, total_costs, initial_capital)
            report_lines.append(f"\n{name.upper()}:")
            for m, v in metrics.items():
                report_lines.append(f"    {m}: {v}")
        
        print("\n".join(report_lines))
        metrics_filepath = os.path.join(run_dir, "metrics_report_v2.txt")
        with open(metrics_filepath, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"\nReporte de metricas guardado en: {metrics_filepath}")
        
        plot_results(results, asset_data, training_window, garch_weights, symbols, run_dir)

    except Exception as e:
        print(f"\nError fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
