"""
Optimización de Carteras con LSTM y Algoritmos Genéticos - VERSIÓN CORREGIDA Y ACTUALIZADA
Para CIMCIA 2025 - UNAM FES Cuautitlán
Correcciones: Alineación de fechas, cálculo de benchmark, validación de LSTM
Actualización: Visualización de la fase de entrenamiento en la gráfica de resultados.
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
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def download_data(symbols, start='2022-06-01', end='2024-01-01'):
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
                print(f"✅ {symbol}: {len(hist)} días descargados")
            else:
                print(f"⚠️  {symbol}: Datos insuficientes ({len(hist)} días)")
        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)[:50]}")
    
    if len(data) < 2:
        print("\n⚠️ Intentando con ETFs de respaldo...")
        backup_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
        for symbol in backup_symbols:
            if symbol not in data:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start, end=end)
                    if len(hist) > 250:
                        data[symbol] = hist['Close']
                        print(f"✅ {symbol}: {len(hist)} días (respaldo)")
                except:
                    continue
    
    if len(data) < 2:
        raise ValueError("No se pudieron descargar suficientes activos")
    
    df = pd.DataFrame(data)
    df = df.dropna()
    
    print(f"\n📊 Dataset final: {len(df)} días, {len(df.columns)} activos")
    print(f"📅 Período: {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}")
    
    return df

def lstm_predict_volatility(prices, lookback=60, validation_split=0.2):
    """LSTM mejorado con validación y regularización"""
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
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
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
    """Algoritmo genético mejorado con matriz de covarianza"""
    n_assets = len(expected_returns)
    population = []
    
    equal_weights = np.ones(n_assets) / n_assets
    population.append(equal_weights)
    
    for i in range(n_assets):
        weights = np.ones(n_assets) * 0.05
        weights[i] = 1 - (n_assets - 1) * 0.05
        population.append(weights)
    
    while len(population) < population_size:
        weights = np.random.dirichlet(np.ones(n_assets))
        population.append(weights)
    
    best_sharpe_history = []
    
    for generation in range(n_generations):
        fitness_scores = []
        for weights in population:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else -np.inf
            concentration_penalty = np.sum(weights**2)
            adjusted_sharpe = sharpe - 0.3 * concentration_penalty
            fitness_scores.append(adjusted_sharpe)
        
        fitness_scores = np.array(fitness_scores)
        best_sharpe_history.append(np.max(fitness_scores))
        
        elite_size = population_size // 4
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        while len(new_population) < population_size:
            tournament_size = 5
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent1 = population[winner_idx]
            
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[winner_idx]
            
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            mutation_rate = 0.1 * (1 - generation / n_generations)
            if np.random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.05, n_assets)
                child = child + mutation
            
            child = np.abs(child)
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
    return population[best_idx], best_sharpe_history

def backtest_strategy(data, lookback_days=252, rebalance_days=30, initial_capital=100000):
    """Backtest con cálculos corregidos y alineación de índices"""
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
        if i >= len(data):
            break
            
        current_date = data.index[i]
        historical_data = data.iloc[:i]
        historical_returns = returns.iloc[:i]
        
        print(f"📅 Rebalanceo en {current_date.strftime('%Y-%m-%d')} (día {i}/{len(data)})")
        
        predicted_vols = []
        for symbol in symbols:
            vol = lstm_predict_volatility(historical_data[symbol])
            predicted_vols.append(vol)
        
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
    
    print(f"\n✅ Backtest completado: {len(portfolio_values)} días simulados")
    
    return (np.array(portfolio_values), np.array(benchmark_values), 
            np.array(weights_history), dates_used, symbols)

def calculate_metrics(values, initial_value=100000):
    """Calcula métricas de rendimiento corregidas"""
    if len(values) < 2:
        return {}
    
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]
    
    total_return = (values[-1] / initial_value) - 1
    
    n_days = len(values)
    years = n_days / 252
    
    annual_return = (values[-1] / initial_value) ** (1/years) - 1 if years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    
    risk_free_rate = 0.02
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    cumulative = values / initial_value
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    return {
        'Retorno Total': f"{total_return:.2%}",
        'Retorno Anual': f"{annual_return:.2%}",
        'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Valor Final': f"${values[-1]:,.0f}"
    }

def plot_results(portfolio_values, benchmark_values, weights_history, dates, symbols, full_data, lookback_days):
    """
    Genera visualizaciones mejoradas con separación de entrenamiento/backtest.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
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
    ax1.axvline(x=separation_date, color='r', linestyle=':', linewidth=2, label='Inicio de Operación')
    y_limit = ax1.get_ylim()
    ax1.text(training_data.index[len(training_data)//4], y_limit[1]*0.95, 'Fase de\nEntrenamiento', ha='center', style='italic', color='gray')
    ax1.text(dates[len(dates)//2], y_limit[1]*0.95, 'Fase de\nBacktest', ha='center', style='italic', color='gray')
    ax1.set_title('Evolución del Valor de Cartera', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Valor ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax2 = axes[0, 1]
    portfolio_cum_returns = (portfolio_values / portfolio_values[0] - 1) * 100
    benchmark_cum_returns = (benchmark_values / benchmark_values[0] - 1) * 100
    ax2.plot(dates, portfolio_cum_returns, label='Cartera LSTM-GA', linewidth=2, color='blue')
    ax2.plot(dates, benchmark_cum_returns, label='Benchmark', linewidth=2, color='orange')
    ax2.set_title('Retornos Acumulados (Durante Backtest)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Retorno (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax3.set_title('Distribución Final de Pesos', fontsize=14, fontweight='bold')
    
    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        weights_array = np.array(weights_history)
        bottom = np.zeros(len(weights_history))
        for i, symbol in enumerate(symbols):
            ax4.bar(range(len(weights_history)), weights_array[:, i], 
                   bottom=bottom, label=symbol, width=1.0)
            bottom += weights_array[:, i]
        ax4.set_title('Evolución de Pesos en el Tiempo', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Rebalanceo #')
        ax4.set_ylabel('Peso (%)')
        ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def main():
    """Función principal con manejo de errores mejorado"""
    print("="*60)
    print("OPTIMIZACIÓN DE CARTERAS CON LSTM Y ALGORITMOS GENÉTICOS")
    print("CIMCIA 2025 - UNAM FES Cuautitlán")
    print("="*60)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    initial_capital = 100000
    lookback_period_days = 252
    
    try:
        data = download_data(symbols, start='2022-01-01', end='2024-01-01')
        results = backtest_strategy(
            data, 
            lookback_days=lookback_period_days,
            rebalance_days=30,
            initial_capital=initial_capital
        )
        portfolio_values, benchmark_values, weights_history, dates, final_symbols = results
        print("\n" + "="*60)
        print("MÉTRICAS DE RENDIMIENTO")
        print("="*60)
        portfolio_metrics = calculate_metrics(portfolio_values, initial_capital)
        benchmark_metrics = calculate_metrics(benchmark_values, initial_capital)
        print("\n📊 CARTERA OPTIMIZADA (LSTM + GA):")
        for metric, value in portfolio_metrics.items():
            print(f"   {metric}: {value}")
        print("\n📊 BENCHMARK (Equal Weight):")
        for metric, value in benchmark_metrics.items():
            print(f"   {metric}: {value}")
        portfolio_sharpe = float(portfolio_metrics['Sharpe Ratio'])
        benchmark_sharpe = float(benchmark_metrics['Sharpe Ratio'])
        if benchmark_sharpe != 0:
            improvement = ((portfolio_sharpe / benchmark_sharpe) - 1) * 100
            print(f"\n🎯 Mejora en Sharpe Ratio: {improvement:.1f}%")
        fig = plot_results(portfolio_values, benchmark_values, weights_history, dates, final_symbols,
                           full_data=data, lookback_days=lookback_period_days)
        plt.show()
        print("\n" + "="*60)
        print("ESTADÍSTICAS ADICIONALES")
        print("="*60)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
        print(f"📈 Correlación con benchmark: {correlation:.3f}")
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        ir = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        print(f"📊 Information Ratio: {ir:.3f}")
        print(f"🔄 Número de rebalanceos: {len(weights_history)}")
        print("\n✅ Análisis completado exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {str(e)}")
        print("\n💡 Sugerencias:")
        print("   1. Verificar conexión a internet para descarga de datos")
        print("   2. Asegurar que los símbolos de acciones sean válidos")
        print("   3. Verificar instalación de librerías requeridas")
        import traceback
        print("\n🔍 Detalle del error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("🎉 PROGRAMA FINALIZADO CON ÉXITO")
    print("="*60)
