"""
Optimización de Carteras con LSTM y Algoritmos Genéticos - VERSIÓN CORREGIDA Y ACTUALIZADA
Para CIMCIA 2025 - UNAM FES Cuautitlán
Correcciones: Alineación de fechas, cálculo de benchmark, validación de LSTM
Actualización: Visualización de la fase de entrenamiento en la gráfica de resultados.
----------------
06/09/2025 - Versión 1.2 - Modificacion en el algoritmo genético para Controlar la Agresividad
                 * Se aumenta la penalización por concentración de pesos
                 * Se limita el peso máximo por activo al 30%
                 * Se agrego guardado de valores y métricas en carpeta con timestamp
06/09/2025 - Versión 1.3 - Modificacion en la funcion de ploteo
                 * Se añade la visualización de la fase de entrenamiento en la gráfica
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
import warnings
import os 
from datetime import datetime 

warnings.filterwarnings('ignore')

# Configuración
np.random.seed(42)
tf.random.set_seed(42)

def download_data(symbols, start, end):
    """Descarga datos con mejor manejo de errores"""
    print("="*60)
    print("DESCARGA DE DATOS")
    print("="*60)
    
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end)
            if len(hist) > 250:
                data[symbol] = hist['Close']
                print(f"Descargados {len(hist)} dias para {symbol}")
            else:
                print(f"Datos insuficientes para {symbol} ({len(hist)} dias)")
        except Exception as e:
            print(f"Error al descargar {symbol}: {str(e)[:50]}")
    
    if len(data) < 2:
        raise ValueError("No se pudieron descargar suficientes activos")
    
    df = pd.DataFrame(data)
    df = df.dropna()
    
    print(f"\nDataset final: {len(df)} dias, {len(df.columns)} activos")
    print(f"Periodo: {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df

def lstm_predict_volatility(prices, lookback=60, validation_split=0.2):
    """LSTM con validación y regularización"""
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
    
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[early_stop],
              verbose=0)
    
    last_sequence = vol_scaled[-lookback:].reshape(1, lookback, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_vol = scaler.inverse_transform(pred_scaled)[0, 0]
    
    return np.clip(pred_vol, 0.05, 0.80)

def genetic_algorithm_optimize(expected_returns, cov_matrix, n_generations=50, population_size=100):
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

def backtest_strategy(data, lookback_days=252, rebalance_days=30, initial_capital=100000):
    print("\n" + "="*60)
    print("EJECUTANDO BACKTEST")
    print("="*60)
    
    symbols = data.columns.tolist()
    n_assets = len(symbols)
    
    if len(data) < lookback_days + rebalance_days:
        raise ValueError("Datos insuficientes para backtest")
    
    portfolio_values = []
    benchmark_values = []
    weights_history = []
    dates_used = []
    
    returns = data.pct_change().fillna(0)
    
    current_portfolio_value = initial_capital
    current_benchmark_value = initial_capital
    
    benchmark_weights = np.ones(n_assets) / n_assets
    
    for i in range(lookback_days, len(data), rebalance_days):
        current_date = data.index[i]
        historical_data = data.iloc[:i]
        historical_returns = returns.iloc[:i]
        
        print(f"Rebalanceo en {current_date.strftime('%Y-%m-%d')} (dia {i}/{len(data)})")
        
        predicted_vols = [lstm_predict_volatility(historical_data[symbol]) for symbol in symbols]
        
        expected_returns = historical_returns.mean().values * 252
        cov_matrix = historical_returns.cov().values * 252
        
        optimal_weights, _ = genetic_algorithm_optimize(
            expected_returns, cov_matrix, 
            n_generations=30, population_size=50
        )
        
        weights_history.append(optimal_weights)
        
        period_end = min(i + rebalance_days, len(data))
        
        for j in range(i, period_end):
            if j > i:
                daily_return = returns.iloc[j]
                portfolio_daily_return = np.dot(optimal_weights, daily_return)
                current_portfolio_value *= (1 + portfolio_daily_return)
                benchmark_daily_return = np.dot(benchmark_weights, daily_return)
                current_benchmark_value *= (1 + benchmark_daily_return)
            
            portfolio_values.append(current_portfolio_value)
            benchmark_values.append(current_benchmark_value)
            dates_used.append(data.index[j])
    
    print(f"\nBacktest completado: {len(portfolio_values)} dias simulados")
    
    return (np.array(portfolio_values), np.array(benchmark_values), 
            np.array(weights_history), dates_used, symbols)

def calculate_metrics(values, initial_value=100000):
    if len(values) < 2: return {}
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]
    total_return = (values[-1] / initial_value) - 1
    years = len(values) / 252
    annual_return = (values[-1] / initial_value) ** (1/years) - 1 if years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    risk_free_rate = 0.02
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    cumulative = values / initial_value
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return {
        'Retorno Total': f"{total_return:.2%}", 'Retorno Anual': f"{annual_return:.2%}",
        'Volatilidad Anual': f"{annual_vol:.2%}", 'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}", 'Valor Final': f"${values[-1]:,.0f}"
    }

def plot_results(portfolio_values, benchmark_values, weights_history, dates, symbols, full_data, lookback_days):
    """
    Genera visualizaciones con separación de entrenamiento/backtest.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Evolución de carteras
    ax1 = axes[0, 0]
    
    separation_date = full_data.index[lookback_days]
    
    initial_capital = benchmark_values[0]
    training_data = full_data.iloc[:lookback_days]
    training_returns = training_data.pct_change().fillna(0).mean(axis=1)
    training_benchmark_values = initial_capital * (1 + training_returns).cumprod()

    num_total_points = len(training_benchmark_values) + len(benchmark_values)
    full_benchmark_dates = full_data.index[:num_total_points]
    full_benchmark_values = np.concatenate([training_benchmark_values.values, benchmark_values])

    ax1.plot(full_benchmark_dates, full_benchmark_values, label='Benchmark (Fase Ent. + Backtest)', linewidth=1.5, color='orange', linestyle='--')
    ax1.plot(dates, portfolio_values, label='Cartera LSTM-GA (Fase Backtest)', linewidth=2, color='blue')
    ax1.axvline(x=separation_date, color='r', linestyle=':', linewidth=2, label='Inicio de Operacion')

    y_limit = ax1.get_ylim()
    ax1.text(training_data.index[len(training_data)//4], y_limit[1]*0.95, 'Fase de\nEntrenamiento', ha='center', style='italic', color='gray')
    ax1.text(dates[len(dates)//2], y_limit[1]*0.95, 'Fase de\nBacktest', ha='center', style='italic', color='gray')
    
    ax1.set_title('Evolucion del Valor de Cartera', fontsize=14)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    portfolio_cum_returns = (portfolio_values / portfolio_values[0] - 1) * 100
    benchmark_cum_returns = (benchmark_values / benchmark_values[0] - 1) * 100
    ax2.plot(dates, portfolio_cum_returns, label='Cartera LSTM-GA', color='blue')
    ax2.plot(dates, benchmark_cum_returns, label='Benchmark', color='orange')
    ax2.set_title('Retornos Acumulados (Durante Backtest)', fontsize=14)
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Retorno (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Distribucion Final de Pesos', fontsize=14)
    
    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        weights_array = np.array(weights_history)
        bottom = np.zeros(len(weights_history))
        for i, symbol in enumerate(symbols):
            ax4.bar(range(len(weights_history)), weights_array[:, i], bottom=bottom, label=symbol)
            bottom += weights_array[:, i]
        ax4.set_title('Evolucion de Pesos en el Tiempo', fontsize=14)
        ax4.set_xlabel('Rebalanceo #')
        ax4.set_ylabel('Peso (%)')
        ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def main():
    print("="*60)
    print("OPTIMIZACION DE CARTERAS CON LSTM Y ALGORITMOS GENETICOS")
    print("="*60)

    years_of_history = 1 
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'INTC']
    start_date = '2021-01-01'
    end_date = '2025-01-01'

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = "results"
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    
    initial_capital = 100000
    lookback_period_days = 252 * years_of_history 
    
    try:
        data = download_data(symbols, start_date, end_date)
        
        results = backtest_strategy(data, lookback_days=lookback_period_days, rebalance_days=30, initial_capital=initial_capital)
        
        portfolio_values, benchmark_values, weights_history, dates, final_symbols = results
        
        portfolio_metrics = calculate_metrics(portfolio_values, initial_capital)
        benchmark_metrics = calculate_metrics(benchmark_values, initial_capital)
        
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("METRICAS DE RENDIMIENTO")
        report_lines.append("="*60)
        
        report_lines.append("\nCARTERA OPTIMIZADA (LSTM + GA):")
        for metric, value in portfolio_metrics.items():
            report_lines.append(f"   {metric}: {value}")
        
        report_lines.append("\nBENCHMARK (Equal Weight):")
        for metric, value in benchmark_metrics.items():
            report_lines.append(f"   {metric}: {value}")
        
        portfolio_sharpe = float(portfolio_metrics.get('Sharpe Ratio', '0'))
        benchmark_sharpe = float(benchmark_metrics.get('Sharpe Ratio', '0'))
        
        if benchmark_sharpe != 0:
            improvement = ((portfolio_sharpe / benchmark_sharpe) - 1) * 100
            report_lines.append(f"\nMejora en Sharpe Ratio: {improvement:.1f}%")

        metrics_filepath = os.path.join(run_dir, "metrics_report.txt")
        with open(metrics_filepath, 'w') as f:
            f.write('\n'.join(report_lines))

        print('\n'.join(report_lines))

        fig = plot_results(portfolio_values, benchmark_values, weights_history, dates, final_symbols,
                           full_data=data, lookback_days=lookback_period_days)
        
        plot_filepath = os.path.join(run_dir, "performance_charts.png")
        fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"\nGrafico guardado en: {plot_filepath}")
        print(f"Reporte de metricas guardado en: {metrics_filepath}")
        
        plt.show()
        
    except Exception as e:
        print(f"\nError en la ejecucion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("PROGRAMA FINALIZADO CON EXITO")
    print("="*60)
