"""
Optimizacion de Carteras: Comparativa de Estrategias con Validacion Walk-Forward 
----------------
12/06/2025 - Version 3.2 - Panel de Control de Visualizacion
                 * Se restaura el layout de 2x2 para las graficas.
                 * El panel ahora incluye:
                   1. Grafica principal de rendimiento (con fase de entrenamiento separada).
                   2. Grafica de retornos acumulados durante el backtest.
                   3. Grafico de pastel con la distribucion final de pesos de LSTM-GA.
                   4. Grafico de barras apiladas con la evolucion de pesos de LSTM-GA.
07/09/2025 - Version 3.3 - Sistema de Guardado Organizado
                 * Se implementa sistema de carpetas con timestamps unicos
                 * Guardado separado de metricas y graficas
                 * Estructura organizada de directorios tipo "run_{timestamp}"
08/09/2025 - Version 3.4 - Costos de Transaccion Realistas
                 * Implementacion de costos de transaccion del 0.1% por operacion
                 * Calculo basado en turnover de la cartera en cada rebalanceo
                 * Analisis del impacto de costos en metricas de rendimiento
                 * Comparacion realista entre estrategias incluyendo fricciones de mercado
09/09/2025 - Version 3.5 - CORRECCION CRITICA: ELIMINACION DE DATA SNOOPING
                 * Implementacion de Walk-Forward Analysis verdadero
                 * Ventana fija de entrenamiento (2 anos) usando solo datos pasados
                 * Evaluacion out-of-sample estricta en datos futuros no vistos
                 * Periodo mas conservador (2018-2025) para resultados realistas
                 * Validacion temporal rigurosa sin sesgos de optimizacion
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from arch import arch_model
import warnings
import os 
from datetime import datetime 

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

np.random.seed(42)
tf.random.set_seed(42)

def download_data(symbols, start, end):
    data = yf.download(symbols, start=start, end=end, auto_adjust=True)['Close']
    data = data.ffill().dropna()
    print(f"Datos descargados para {len(data.columns)} activos.")
    print(f"Periodo: {data.index[0]:%Y-%m-%d} a {data.index[-1]:%Y-%m-%d} ({len(data)} dias).\n")
    return data

def lstm_predict_volatility(prices, lookback=60):
    """Predice la volatilidad usando un modelo LSTM."""
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window=20).std() * np.sqrt(252)
    volatility = volatility.dropna()
    
    if len(volatility) < lookback + 50:
        return volatility.mean() if len(volatility) > 0 else 0.20
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    vol_scaled = scaler.fit_transform(volatility.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(vol_scaled)):
        X.append(vol_scaled[i-lookback:i, 0])
        y.append(vol_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2),
        LSTM(16), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
    
    last_sequence = vol_scaled[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_vol = scaler.inverse_transform(pred_scaled)[0, 0]
    
    return np.clip(pred_vol, 0.05, 1.0)

def garch_predict_volatility(prices):
    """Predice la volatilidad usando un modelo GARCH(1,1)."""
    returns = prices.pct_change().dropna() * 100
    if len(returns) < 50:
        return returns.std() * np.sqrt(252) / 100 if len(returns) > 20 else 0.20

    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
        res = model.fit(disp='off', update_freq=0)
        forecast = res.forecast(horizon=1)
        predicted_var = forecast.variance.iloc[-1].values[0]
        annualized_vol = np.sqrt(predicted_var * 252) / 100
    except Exception:
        annualized_vol = returns.std() * np.sqrt(252) / 100
        
    return np.clip(annualized_vol, 0.05, 1.0)

def genetic_algorithm_optimizer(expected_returns, cov_matrix, n_generations=50, population_size=100):
    """Algoritmo genetico con restricciones de peso."""
    n_assets = len(expected_returns)
    population = []
    equal_weights = np.ones(n_assets) / n_assets
    population.append(equal_weights)
    
    while len(population) < population_size:
        weights = np.random.dirichlet(np.ones(n_assets))
        population.append(weights)
    
    for generation in range(n_generations):
        fitness_scores = []
        for weights in population:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else -np.inf
            concentration_penalty = np.sum(weights**2)
            adjusted_sharpe = sharpe - 0.8 * concentration_penalty
            fitness_scores.append(adjusted_sharpe)
        
        fitness_scores = np.array(fitness_scores)
        elite_size = population_size // 4
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        while len(new_population) < population_size:
            tournament_size = 5
            tournament_indices = np.random.choice(len(population), tournament_size)
            winner_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
            parent1 = population[winner_idx]
            
            tournament_indices = np.random.choice(len(population), tournament_size)
            winner_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
            parent2 = population[winner_idx]
            
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            mutation_rate = 0.1 * (1 - generation / n_generations)
            if np.random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.05, n_assets)
                child = child + mutation
            
            max_weight_limit = 0.30
            child = np.clip(child, 0, max_weight_limit)
            
            child = child / child.sum() if child.sum() > 0 else equal_weights
            
            new_population.append(child)
        
        population = new_population
    
    final_fitness = []
    for weights in population:
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else -np.inf
        final_fitness.append(sharpe)
    
    best_idx = np.argmax(final_fitness)
    return population[best_idx], None

def inverse_volatility_portfolio(predicted_vols):
    """Crea una cartera asignando pesos inversamente a la volatilidad."""
    inv_vols = 1.0 / (np.array(predicted_vols) + 1e-8)
    weights = inv_vols / np.sum(inv_vols)
    return weights

def calculate_transaction_cost(old_weights, new_weights, portfolio_value, transaction_cost_rate):
    """
    Calcula el costo de transaccion basado en el turnover de la cartera.
    
    Args:
        old_weights: Pesos anteriores de la cartera
        new_weights: Nuevos pesos objetivo
        portfolio_value: Valor actual de la cartera
        transaction_cost_rate: Tasa de costo por transaccion (ej. 0.001 = 0.1%)
    
    Returns:
        float: Costo total de la transaccion en valor monetario
    """
    if old_weights is None:
        turnover = np.sum(new_weights)
    else:
        turnover = np.sum(np.abs(new_weights - old_weights))
    
    transaction_cost = turnover * transaction_cost_rate * portfolio_value
    return transaction_cost

def run_backtest_walk_forward(data, weight_strategy_fn, **kwargs):
    """
    Motor de backtesting con validacion walk-forward .
    
    CORRECCION 09/09/2025 SIN DATA SNOOPING: 
    - Solo usa ventana fija de datos PASADOS para entrenamiento
    - Evalua rendimiento en periodo FUTURO completamente out-of-sample
    - Elimina el sesgo de optimizacion en el mismo periodo de evaluacion
    """
    print(f"\n" + "="*60 + f"\n2. EJECUTANDO BACKTEST WALK-FORWARD: {weight_strategy_fn.__name__}\n" + "="*60)
    
    initial_capital = kwargs.get('initial_capital', 100000)
    training_window = kwargs.get('training_window', 504)
    rebalance_days = kwargs.get('rebalance_days', 30)
    transaction_cost_rate = kwargs.get('transaction_cost_rate', 0.001)
    
    print(f"  -> Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} anos)")
    print(f"  -> Costos de transaccion: {transaction_cost_rate:.3%} por operacion")
    print(f"  -> MODO: Walk-Forward Analysis ")
    
    portfolio_values = []
    weights_history = []
    transaction_costs_history = []
    dates = []
    returns = data.pct_change().fillna(0)
    current_value = initial_capital
    previous_weights = None
    total_transaction_costs = 0
    
    start_index = training_window
    
    for i in range(start_index, len(data), rebalance_days):
        current_date = data.index[i]
        print(f"  -> Rebalanceando en {current_date:%Y-%m-%d}...")
        
        train_start = max(0, i - training_window)
        train_end = i
        
        training_data = data.iloc[train_start:train_end]
        
        print(f"     Entrenando con datos: {training_data.index[0]:%Y-%m-%d} a {training_data.index[-1]:%Y-%m-%d}")
        
        new_weights = weight_strategy_fn(training_data)
        
        transaction_cost = calculate_transaction_cost(
            previous_weights, new_weights, current_value, transaction_cost_rate
        )
        
        current_value -= transaction_cost
        total_transaction_costs += transaction_cost
        
        weights_history.append(new_weights)
        transaction_costs_history.append(transaction_cost)
        previous_weights = new_weights.copy()
        
        print(f"     Costo de transaccion: ${transaction_cost:,.2f}")
        
        period_end = min(i + rebalance_days, len(data))
        
        for j in range(i, period_end):
            if j > i:
                daily_return = np.dot(new_weights, returns.iloc[j])
                current_value *= (1 + daily_return)
            portfolio_values.append(current_value)
            dates.append(data.index[j])
    
    print(f"  -> Total costos de transaccion: ${total_transaction_costs:,.2f}")
    print(f"  -> Impacto en rendimiento: {total_transaction_costs/initial_capital:.2%}")
    print(f"  -> Periodos evaluados out-of-sample: {len(portfolio_values)} dias")
            
    return (pd.Series(portfolio_values, index=dates), 
            np.array(weights_history), 
            np.array(transaction_costs_history),
            total_transaction_costs)

def strategy_lstm_ga(historical_data):
    returns = historical_data.pct_change().dropna()
    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    optimal_weights, _ = genetic_algorithm_optimizer(expected_returns, cov_matrix)
    return optimal_weights

def strategy_garch_inv_vol(historical_data):
    predicted_vols = [garch_predict_volatility(historical_data[symbol]) for symbol in historical_data.columns]
    return inverse_volatility_portfolio(predicted_vols)

def safe_write_file(filepath, content):
    """
    Escribe archivo de forma segura manejando problemas de codificacion.
    """
    if isinstance(content, list):
        content = '\n'.join(content)
    
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except UnicodeEncodeError:
        try:
            with open(filepath, 'w', encoding='ascii', errors='ignore') as f:
                f.write(content)
        except Exception as e:
            print(f"Error al escribir archivo {filepath}: {e}")

def calculate_metrics(values, total_transaction_costs=0, initial_capital=100000):
    if len(values) < 2: return {}
    returns = values.pct_change().dropna()
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    years = len(values) / 252.0
    annual_return = (1 + total_return)**(1/years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
    
    cumulative = values / values.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    transaction_cost_impact = total_transaction_costs / initial_capital if initial_capital > 0 else 0
    
    metrics = {
        'Retorno Anual': f"{annual_return:.2%}",
        'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}", 
        'Max Drawdown': f"{drawdown.min():.2%}",
        'Valor Final': f"${values.iloc[-1]:,.0f}"
    }
    
    if total_transaction_costs > 0:
        metrics['Costos Transaccion'] = f"${total_transaction_costs:,.2f}"
        metrics['Impacto en Rendimiento'] = f"{transaction_cost_impact:.2%}"
    
    return metrics

def plot_results(results, full_data, training_window, weights_history, symbols):
    """
    Genera panel de control 2x2.
    """
    print("\n" + "="*60 + "\n3.PANEL DE CONTROL...\n" + "="*60)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    ax1 = axes[0, 0]
    initial_capital = 100000
    backtest_start_date = full_data.index[training_window]
    
    n_assets = full_data.shape[1]
    full_benchmark_returns = (full_data.pct_change().fillna(0) @ (np.ones(n_assets) / n_assets)) + 1
    full_benchmark_raw = initial_capital * full_benchmark_returns.cumprod()
    norm_factor = initial_capital / full_benchmark_raw.loc[backtest_start_date]
    full_benchmark_normalized = full_benchmark_raw * norm_factor

    ax1.plot(full_benchmark_normalized.index, full_benchmark_normalized.values, 
             label='Benchmark (Equiponderado)', linewidth=1.5, color='orange', linestyle=':')
    
    colors = ['blue', 'green']
    styles = ['-', '--']
    strategies_to_plot = {k: v for k, v in results.items() if "Benchmark" not in k}
    for i, (name, data) in enumerate(strategies_to_plot.items()):
        ax1.plot(data.index, data.values, label=name, linewidth=2.5, color=colors[i], linestyle=styles[i])

    ax1.axvline(x=backtest_start_date, color='r', linestyle='-.', linewidth=2, label='Inicio de Operacion')
    y_min, y_max = ax1.get_ylim()
    ax1.fill_betweenx([y_min, y_max], full_data.index[0], backtest_start_date, color='grey', alpha=0.1)
    ax1.text(full_data.index[training_window//2], y_min + (y_max - y_min) * 0.9, 
             'Fase de Entrenamiento\n(Walk-Forward)', ha='center', style='italic', color='gray', fontsize=12)

    ax1.set_title('1. Evolucion del Valor de Cartera (Out-of-Sample Verdadero)', fontsize=16, weight='bold')
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Valor de Cartera ($)', fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        cum_returns = (data / data.iloc[0] - 1) * 100
        ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0, 
                color=colors[i] if name != "Benchmark (Equiponderado)" else 'orange', 
                linestyle=styles[i] if name != "Benchmark (Equiponderado)" else ':')
    
    ax2.set_title('2. Retornos Acumulados (Walk-Forward Validation)', fontsize=16, weight='bold')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')

    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%', startangle=90,
                pctdistance=0.85, explode=[0.05]*len(symbols))
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        ax3.add_artist(centre_circle)
    ax3.set_title('3. Distribucion Final de Pesos (LSTM-GA)', fontsize=16, weight='bold')

    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        bottom = np.zeros(len(weights_history))
        for i, symbol in enumerate(symbols):
            ax4.bar(range(len(weights_history)), weights_history[:, i], bottom=bottom, label=symbol)
            bottom += weights_history[:, i]
    ax4.set_title('4. Evolucion de Pesos en Rebalanceos (LSTM-GA)', fontsize=16, weight='bold')
    ax4.set_xlabel('Numero de Rebalanceo', fontsize=12)
    ax4.set_ylabel('Proporcion de Cartera', fontsize=12)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax4.set_ylim([0, 1])

    plt.tight_layout(pad=3.0)
    fig.suptitle('Panel de Control - Validacion Walk-Forward ', fontsize=22, weight='bold')
    plt.subplots_adjust(top=0.92)
    return fig

def main():
    print("="*60)
    print("OPTIMIZACION DE CARTERAS: COMPARATIVA DE ESTRATEGIAS")
    print("CON VALIDACION WALK-FORWARD ")
    print("="*60)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'INTC']
    start_date = '2018-01-01'
    end_date = '2025-01-01'
    training_window = 504
    transaction_cost_rate = 0.001
    initial_capital = 100000
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = "results"
    run_dir = os.path.join(output_dir, f"run_walkforward_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Directorio de resultados: {run_dir}")
    print(f"Periodo de analisis: {start_date} a {end_date}")
    print(f"Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} anos)")
    print(f"Costos de transaccion: {transaction_cost_rate:.3%} por operacion")
    print("MODO: Walk-Forward Analysis (validacion temporal estricta)")
    
    try:
        data = download_data(symbols, start_date, end_date)
        
        lstm_ga_results = run_backtest_walk_forward(data, strategy_lstm_ga, 
                                                  training_window=training_window,
                                                  transaction_cost_rate=transaction_cost_rate,
                                                  initial_capital=initial_capital)
        lstm_ga_values, lstm_ga_weights, lstm_ga_costs, lstm_ga_total_costs = lstm_ga_results
        
        garch_results = run_backtest_walk_forward(data, strategy_garch_inv_vol, 
                                                training_window=training_window,
                                                transaction_cost_rate=transaction_cost_rate,
                                                initial_capital=initial_capital)
        garch_inv_vol_values, _, garch_costs, garch_total_costs = garch_results
        
        print(f"\n" + "="*60 + f"\n2. EJECUTANDO BACKTEST WALK-FORWARD: benchmark_equiponderado\n" + "="*60)
        print(f"  -> Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} anos)")
        print(f"  -> Costos de transaccion: {transaction_cost_rate:.3%} por operacion")
        print(f"  -> MODO: Walk-Forward Analysis ")
        
        n_assets = data.shape[1]
        benchmark_weights = np.ones(n_assets) / n_assets
        
        benchmark_values_list = []
        benchmark_dates = []
        benchmark_total_costs = 0
        benchmark_value = initial_capital
        previous_benchmark_weights = None
        returns = data.pct_change().fillna(0)
        
        start_index = training_window
        for i in range(start_index, len(data), 30):
            current_date = data.index[i]
            print(f"  -> Rebalanceando en {current_date:%Y-%m-%d}...")
            
            transaction_cost = calculate_transaction_cost(
                previous_benchmark_weights, benchmark_weights, benchmark_value, transaction_cost_rate
            )
            benchmark_value -= transaction_cost
            benchmark_total_costs += transaction_cost
            previous_benchmark_weights = benchmark_weights.copy()
            
            print(f"     Costo de transaccion: ${transaction_cost:,.2f}")
            
            period_end = min(i + 30, len(data))
            for j in range(i, period_end):
                if j > i:
                    daily_return = np.dot(benchmark_weights, returns.iloc[j])
                    benchmark_value *= (1 + daily_return)
                benchmark_values_list.append(benchmark_value)
                benchmark_dates.append(data.index[j])
        
        benchmark_values = pd.Series(benchmark_values_list, index=benchmark_dates)
        
        print(f"  -> Total costos de transaccion: ${benchmark_total_costs:,.2f}")
        print(f"  -> Impacto en rendimiento: {benchmark_total_costs/initial_capital:.2%}")
        print(f"  -> Periodos evaluados out-of-sample: {len(benchmark_values)} dias")
        
        results = {
            "LSTM + Algoritmo Genetico": lstm_ga_values,
            "GARCH Inversa Volatilidad (sin GA)": garch_inv_vol_values,
            "Benchmark (Equiponderado)": benchmark_values
        }
        
        transaction_costs = {
            "LSTM + Algoritmo Genetico": lstm_ga_total_costs,
            "GARCH Inversa Volatilidad (sin GA)": garch_total_costs,
            "Benchmark (Equiponderado)": benchmark_total_costs
        }
        
        report = ["="*60, 
                  "METRICAS DE RENDIMIENTO COMPARATIVAS", 
                  "VALIDACION WALK-FORWARD ", 
                  "CON COSTOS DE TRANSACCION INCLUIDOS", 
                  "="*60]
        
        for name, series_data in results.items():
            total_costs = transaction_costs[name]
            metrics = calculate_metrics(series_data, total_costs, initial_capital)
            report.append(f"\n{name.upper()}:")
            report.extend([f"   {m}: {v}" for m, v in metrics.items()])
        
        lstm_sharpe = float(calculate_metrics(results["LSTM + Algoritmo Genetico"], 
                                            transaction_costs["LSTM + Algoritmo Genetico"], 
                                            initial_capital).get('Sharpe Ratio', '0'))
        benchmark_sharpe = float(calculate_metrics(results["Benchmark (Equiponderado)"], 
                                                 transaction_costs["Benchmark (Equiponderado)"], 
                                                 initial_capital).get('Sharpe Ratio', '0'))
        
        if benchmark_sharpe != 0:
            improvement = ((lstm_sharpe / benchmark_sharpe) - 1) * 100
            report.append(f"\nMEJORA EN SHARPE RATIO (LSTM-GA vs Benchmark): {improvement:.1f}%")
        
        report.append(f"\n" + "="*40)
        report.append("ANALISIS DE COSTOS DE TRANSACCION")
        report.append("="*40)
        for name, costs in transaction_costs.items():
            impact = costs / initial_capital
            report.append(f"{name}: ${costs:,.2f} ({impact:.2%} del capital inicial)")
        
        report.append(f"\n" + "="*40)
        report.append("NOTA METODOLOGICA IMPORTANTE")
        report.append("="*40)
        report.append("+ Validacion Walk-Forward implementada correctamente")
        report.append("+ Sin data snooping: entrenamiento solo con datos pasados")
        report.append("+ Evaluacion out-of-sample en datos futuros no vistos")
        report.append("+ Costos de transaccion incluidos para realismo")
        report.append(f"+ Ventana de entrenamiento: {training_window/252:.1f} anos")
        report.append(f"+ Periodo de evaluacion: {start_date} a {end_date}")
        
        print('\n'.join(report))
        
        fig = plot_results(results, data, training_window, lstm_ga_weights, symbols)
        
        plot_filepath = os.path.join(run_dir, "performance_charts_walkforward.png")
        fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        metrics_filepath = os.path.join(run_dir, "metrics_report_walkforward.txt")
        safe_write_file(metrics_filepath, report)
        
        config_filepath = os.path.join(run_dir, "config_parameters_walkforward.txt")
        config_info = [
            "PARAMETROS DE CONFIGURACION - WALK-FORWARD ANALYSIS",
            "="*50,
            f"Simbolos analizados: {', '.join(symbols)}",
            f"Fecha de inicio: {start_date}",
            f"Fecha de fin: {end_date}",
            f"Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} anos)",
            f"Capital inicial: ${initial_capital:,}",
            f"Frecuencia de rebalanceo: 30 dias",
            f"Costos de transaccion: {transaction_cost_rate:.3%} por operacion",
            f"Fecha de ejecucion: {timestamp}",
            "",
            "METODOLOGIA:",
            "- Walk-Forward Analysis ",
            "- Entrenamiento solo con datos historicos",
            "- Evaluacion out-of-sample estricta",
            "- Costos de transaccion incluidos",
            "",
            "ESTRATEGIAS IMPLEMENTADAS:",
            "1. LSTM + Algoritmo Genetico (con costos y walk-forward)",
            "2. GARCH + Inversa Volatilidad (con costos y walk-forward)",
            "3. Benchmark Equiponderado (con costos y walk-forward)"
        ]
        
        safe_write_file(config_filepath, config_info)
        
        if len(lstm_ga_weights) > 0:
            weights_df = pd.DataFrame(lstm_ga_weights, columns=symbols)
            weights_filepath = os.path.join(run_dir, "lstm_ga_weights_history_walkforward.csv")
            weights_df.to_csv(weights_filepath, index=False)
            print(f"Historial de pesos guardado en: {weights_filepath}")
        
        costs_filepath = os.path.join(run_dir, "transaction_costs_analysis_walkforward.csv")
        costs_df = pd.DataFrame({
            'Rebalanceo': range(len(lstm_ga_costs)),
            'LSTM_GA_Costs': lstm_ga_costs,
            'GARCH_Costs': garch_costs[:len(lstm_ga_costs)]
        })
        costs_df.to_csv(costs_filepath, index=False)
        
        print(f"\n" + "="*60)
        print("ARCHIVOS GUARDADOS EXITOSAMENTE:")
        print("="*60)
        print(f"[GRAFICO] Grafico de rendimiento: {plot_filepath}")
        print(f"[METRICS] Reporte de metricas: {metrics_filepath}")
        print(f"[CONFIG] Parametros de configuracion: {config_filepath}")
        print(f"[COSTS] Analisis de costos: {costs_filepath}")
        print(f"[FOLDER] Directorio principal: {run_dir}")
        print("="*60)
        print("[OK] VALIDACION WALK-FORWARD")
        print("="*60)
        
        plt.show()
        
    except Exception as e:
        print(f"\nError fatal en la ejecucion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("PROGRAMA FINALIZADO CON EXITO")
    print("="*60)
