"""
Optimización de Carteras: Comparativa de Estrategias - VERSIÓN FINAL
Para CIMCIA 2025 - UNAM FES Cuautitlán
Actualización: Se añade una estrategia de Inversa Volatilidad basada únicamente en GARCH,
sin el uso de Algoritmos Genéticos (GA), para una comparación más robusta.
----------------
08/09/2025 - Versión 3.2 - Panel de Control de Visualización
                 * Se restaura el layout de 2x2 para las gráficas.
                 * El panel ahora incluye:
                   1. Gráfica principal de rendimiento (con fase de entrenamiento separada).
                   2. Gráfica de retornos acumulados durante el backtest.
                   3. Gráfico de pastel con la distribución final de pesos de LSTM-GA.
                   4. Gráfico de barras apiladas con la evolución de pesos de LSTM-GA.
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


# --- Configuración General ---
np.random.seed(42)
tf.random.set_seed(42)

# ==============================================================================
# 1. MÓDULO DE ADQUISICIÓN DE DATOS
# ==============================================================================
def download_data(symbols, start, end):
    print("="*60 + "\n1. DESCARGANDO DATOS...\n" + "="*60)
    data = yf.download(symbols, start=start, end=end, auto_adjust=True)['Close']
    data = data.ffill().dropna()
    print(f"Datos descargados para {len(data.columns)} activos.")
    print(f"Periodo: {data.index[0]:%Y-%m-%d} a {data.index[-1]:%Y-%m-%d} ({len(data)} días).\n")
    return data

# ==============================================================================
# 2. MÓDULOS DE PREDICCIÓN DE VOLATILIDAD
# ==============================================================================
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
    
    return np.clip(pred_vol, 0.05, 1.0) # Clip para evitar valores extremos

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
        annualized_vol = returns.std() * np.sqrt(252) / 100 # Fallback
        
    return np.clip(annualized_vol, 0.05, 1.0)

# ==============================================================================
# 3. MÓDULOS DE CONSTRUCCIÓN DE CARTERA
# ==============================================================================
def genetic_algorithm_optimizer(expected_returns, cov_matrix, n_generations=50, population_size=100):
    """Algoritmo genético con restricciones de peso."""
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
            adjusted_sharpe = sharpe - 0.8 * concentration_penalty # Penalización 
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

# ==============================================================================
# 4. MÓDULO DE BACKTESTING
# ==============================================================================
def run_backtest(data, weight_strategy_fn, **kwargs):
    """Motor de backtesting genérico que ahora devuelve el historial de pesos."""
    print(f"\n" + "="*60 + f"\n2. EJECUTANDO BACKTEST: {weight_strategy_fn.__name__}\n" + "="*60)
    
    initial_capital = kwargs.get('initial_capital', 100000)
    lookback_days = kwargs.get('lookback_days', 252)
    rebalance_days = kwargs.get('rebalance_days', 30)
    
    portfolio_values = []
    weights_history = []
    dates = []
    returns = data.pct_change().fillna(0)
    current_value = initial_capital
    
    for i in range(lookback_days, len(data), rebalance_days):
        print(f"  -> Rebalanceando en {data.index[i]:%Y-%m-%d}...")
        historical_data = data.iloc[:i]
        weights = weight_strategy_fn(historical_data)
        weights_history.append(weights)
        
        period_end = min(i + rebalance_days, len(data))
        for j in range(i, period_end):
            if j > i:
                daily_return = np.dot(weights, returns.iloc[j])
                current_value *= (1 + daily_return)
            portfolio_values.append(current_value)
            dates.append(data.index[j])
            
    return pd.Series(portfolio_values, index=dates), np.array(weights_history)

# --- Funciones específicas para cada estrategia ---
def strategy_lstm_ga(historical_data):
    returns = historical_data.pct_change().dropna()
    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    # La nueva función GA devuelve una tupla (pesos, None), solo necesitamos los pesos.
    optimal_weights, _ = genetic_algorithm_optimizer(expected_returns, cov_matrix)
    return optimal_weights

def strategy_garch_inv_vol(historical_data):
    predicted_vols = [garch_predict_volatility(historical_data[symbol]) for symbol in historical_data.columns]
    return inverse_volatility_portfolio(predicted_vols)

# ==============================================================================
# 5. MÓDULO DE ANÁLISIS Y VISUALIZACIÓN
# ==============================================================================
def calculate_metrics(values):
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
    
    return {
        'Retorno Anual': f"{annual_return:.2%}",'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}", 'Max Drawdown': f"{drawdown.min():.2%}",
        'Valor Final': f"${values.iloc[-1]:,.0f}"
    }

def plot_results(results, full_data, lookback_days, weights_history, symbols):
    """
    MODIFICADO: Genera un panel de control 2x2 con análisis detallado.
    """
    print("\n" + "="*60 + "\n3. GENERANDO PANEL DE CONTROL...\n" + "="*60)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # --- Gráfica 1: Evolución del Valor de la Cartera (Principal) ---
    ax1 = axes[0, 0]
    initial_capital = 100000
    backtest_start_date = full_data.index[lookback_days]
    
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

    ax1.axvline(x=backtest_start_date, color='r', linestyle='-.', linewidth=2, label='Inicio de Operación')
    y_min, y_max = ax1.get_ylim()
    ax1.fill_betweenx([y_min, y_max], full_data.index[0], backtest_start_date, color='grey', alpha=0.1)
    ax1.text(full_data.index[lookback_days//2], y_min + (y_max - y_min) * 0.9, 
             'Fase de Entrenamiento', ha='center', style='italic', color='gray', fontsize=12)

    ax1.set_title('1. Evolución del Valor de Cartera (Out-of-Sample)', fontsize=16, weight='bold')
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Valor de Cartera ($)', fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # --- Gráfica 2: Retornos Acumulados ---
    ax2 = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        cum_returns = (data / data.iloc[0] - 1) * 100
        ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0, color=colors[i] if name != "Benchmark (Equiponderado)" else 'orange', linestyle=styles[i] if name != "Benchmark (Equiponderado)" else ':')
    
    ax2.set_title('2. Retornos Acumulados (Durante Backtest)', fontsize=16, weight='bold')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')

    # --- Gráfica 3: Distribución Final de Pesos (Pie Chart) ---
    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%', startangle=90,
                pctdistance=0.85, explode=[0.05]*len(symbols))
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        ax3.add_artist(centre_circle)
    ax3.set_title('3. Distribución Final de Pesos (LSTM-GA)', fontsize=16, weight='bold')

    # --- Gráfica 4: Evolución de Pesos ---
    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        bottom = np.zeros(len(weights_history))
        for i, symbol in enumerate(symbols):
            ax4.bar(range(len(weights_history)), weights_history[:, i], bottom=bottom, label=symbol)
            bottom += weights_history[:, i]
    ax4.set_title('4. Evolución de Pesos en Rebalanceos (LSTM-GA)', fontsize=16, weight='bold')
    ax4.set_xlabel('Número de Rebalanceo', fontsize=12)
    ax4.set_ylabel('Proporción de Cartera', fontsize=12)
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax4.set_ylim([0, 1])

    plt.tight_layout(pad=3.0)
    fig.suptitle('Panel de Control de Rendimiento de Estrategias Cuantitativas', fontsize=22, weight='bold')
    plt.subplots_adjust(top=0.92)
    return fig

# ==============================================================================
# 6. ORQUESTADOR PRINCIPAL
# ==============================================================================
def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'INTC']
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    lookback = 252 # 1 año
    
    try:
        data = download_data(symbols, start_date, end_date)
        
        # --- Ejecutar los 3 backtests ---
        lstm_ga_values, lstm_ga_weights = run_backtest(data, strategy_lstm_ga, lookback_days=lookback)
        garch_inv_vol_values, _ = run_backtest(data, strategy_garch_inv_vol, lookback_days=lookback)
        
        # Benchmark Equiponderado (solo para el período de backtest para las métricas)
        n_assets = data.shape[1]
        benchmark_returns = (data.pct_change().fillna(0)[lookback:] @ (np.ones(n_assets) / n_assets)) + 1
        benchmark_values = 100000 * benchmark_returns.cumprod()
        
        results = {
            "LSTM + Algoritmo Genético": lstm_ga_values,
            "GARCH Inversa Volatilidad (sin GA)": garch_inv_vol_values,
            "Benchmark (Equiponderado)": benchmark_values
        }
        
        # --- Calcular y mostrar métricas ---
        report = ["="*60, "MÉTRICAS DE RENDIMIENTO COMPARATIVAS (PERÍODO DE BACKTEST)", "="*60]
        for name, series_data in results.items():
            metrics = calculate_metrics(series_data)
            report.append(f"\n{name.upper()}:")
            report.extend([f"   {m}: {v}" for m, v in metrics.items()])
        
        print('\n'.join(report))
        
        # --- Graficar y guardar resultados ---
        fig = plot_results(results, data, lookback, lstm_ga_weights, symbols)
        
        run_dir = "results"
        os.makedirs(run_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(run_dir, f"comparativa_{timestamp}.png")
        report_path = os.path.join(run_dir, f"reporte_{timestamp}.txt")
        
        fig.savefig(plot_path, dpi=300)
        with open(report_path, 'w') as f: f.write('\n'.join(report))
        
        print(f"\nGráfico guardado en: {plot_path}")
        print(f"Reporte guardado en: {report_path}\n" + "="*60)
        
        plt.show()
        
    except Exception as e:
        print(f"\nError fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


