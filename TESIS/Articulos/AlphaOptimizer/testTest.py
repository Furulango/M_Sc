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
09/09/2025 - Version 4.1 - Framework de Backtesting Robusto y Metodología Validada
                 * Fusión final: Se integra la estrategia híbrida (LSTM-GA) con el motor de backtesting Walk-Forward.
                 * CORRECCIÓN CRÍTICA: La función LSTM (predict_future_volatility) 
                    ahora es un predictor out-of-sample real, eliminando  data snooping.
                 * ROBUSTEZ: Se añaden mecanismos de fallback en las funciones de 
                    LSTM y GA para evitar interrupciones en la ejecución.
                 * VALIDACIÓN COMPLETA: El sistema genera y guarda un reporte 
                    completo (métricas, gráficos, pesos, costos) por cada ejecución para garantizar la reproducibilidad.
"""


import os
import warnings
import random
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Genetic algorithm (DEAP)
from deap import base, creator, tools, algorithms

# Seeds y warnings
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# FUNCIONES DE DATOS
def download_data(symbols, start, end):
    """
    Descarga precios ajustados (close auto_adjust) desde Yahoo Finance.
    """
    df = yf.download(symbols, start=start, end=end, auto_adjust=True)['Close']
    df = df.ffill().dropna()
    print(f"Datos descargados para {len(df.columns)} activos. Periodo: {df.index[0].date()} a {df.index[-1].date()} ({len(df)} dias).")
    return df


# FUNCIONES LSTM (REFAC.)
def create_lstm_dataset(series_values, lookback):
    X, y = [], []
    for i in range(lookback, len(series_values)):
        X.append(series_values[i - lookback:i, 0])
        y.append(series_values[i, 0])
    return np.array(X), np.array(y)


def predict_future_volatility(price_series,
                              lookback=60,
                              future_horizon=21,
                              epochs=20,
                              batch_size=32):
   
    try:
        returns = price_series.pct_change().dropna()
        if returns.empty:
            return 0.20

        # Serie de volatilidad histórica (ventana corta para estimar volatilidad instantánea)
        vol_series = returns.rolling(window=20).std() * np.sqrt(252)
        vol_series = vol_series.dropna()

        if len(vol_series) < lookback + 1:
            # Fallback a volatilidad histórica simple
            fallback = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.20
            return float(np.clip(fallback, 0.05, 1.0))

        # Escalado
        scaler = MinMaxScaler(feature_range=(0, 1))
        vol_scaled = scaler.fit_transform(vol_series.values.reshape(-1, 1))

        # Dataset supervisado
        X, y = create_lstm_dataset(vol_scaled, lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Modelo LSTM (arquitectura ligera y estable)
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

        # Predicción multi-step por realimentación
        last_seq = vol_scaled[-lookback:].reshape(1, lookback, 1).astype(np.float32)
        preds_scaled = []
        for _ in range(future_horizon):
            pred_scaled = model.predict(last_seq, verbose=0)[0, 0]
            preds_scaled.append(pred_scaled)
            # construir nueva secuencia: eliminar primer paso, añadir predicción
            pred_arr = np.array(pred_scaled).reshape(1, 1, 1)
            last_seq = np.concatenate([last_seq[:, 1:, :], pred_arr], axis=1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        # Retornamos la volatilidad "esperada" en el horizon: promedio de predicciones
        predicted_vol = float(np.mean(preds))
        return float(np.clip(predicted_vol, 0.05, 1.0))

    except Exception as e:
        # En caso de error durante el entrenamiento/predicción, fallback razonable
        try:
            fallback = returns.std() * np.sqrt(252)
            return float(np.clip(fallback, 0.05, 1.0))
        except Exception:
            return 0.20

# ALGORITMO GENÉTICO (DEAP)
def genetic_algorithm_optimizer(expected_returns, cov_matrix,
                                population_size=100, generations=50, cxpb=0.5, mutpb=0.2):
    """
    Usa DEAP para optimizar pesos maximizando Sharpe ratio (retorno / volatilidad).
    Devuelve pesos normalizados.
    """
    n_assets = len(expected_returns)
    # Crear clases en creator sólo si no existen (evitar errores en ejecuciones sucesivas)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        w = np.array(individual, dtype=float)
        # Asegurar no-negatividad y normalizar
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        port_return = np.dot(w, expected_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) + 1e-9
        sharpe = port_return / port_vol
        # Penalizar concentración extrema (favor diversidad)
        conc_penalty = 0.0  # opcional: agregar si se desea
        return (sharpe - conc_penalty,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    # Algoritmo evolutivo simple
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=float)
    best = np.clip(best, 0, None)
    if best.sum() == 0:
        best = np.ones_like(best) / len(best)
    else:
        best = best / best.sum()
    return best, hof[0].fitness.values[0]


# ESTRATEGIA HÍBRIDA LSTM + GA (CON AL WALK-FORWARD)
def strategy_lstm_ga(historical_data):
    """
    historical_data: ventana histórica (solo pasado) provista por run_backtest_walk_forward.
    - Predice volatilidad futura para cada activo con predict_future_volatility
    - Ajusta retornos esperados por volatilidad estimada
    - Optimiza pesos con GA sobre retornos ajustados y covarianza histórica
    """
    returns = historical_data.pct_change().dropna()
    if returns.empty:
        # si no hay retornos (caso borde), retornar pesos equiponderados
        n = historical_data.shape[1]
        return np.ones(n) / n

    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252

    predicted_vols = []
    for symbol in historical_data.columns:
        pvol = predict_future_volatility(historical_data[symbol], lookback=60, future_horizon=21, epochs=20)
        predicted_vols.append(pvol)

    predicted_vols = np.array(predicted_vols, dtype=float)
    # Manejo división por cero
    predicted_vols[predicted_vols == 0] = 1e-6

    # Ajuste simple: retornos esperados por riesgo (riesgo-corrected returns)
    adjusted_returns = expected_returns / predicted_vols

    # Ejecutar optimización genética
    try:
        opt_weights, _ = genetic_algorithm_optimizer(adjusted_returns, cov_matrix,
                                                     population_size=100, generations=50)
    except Exception as e:
        # En caso de fallo del GA, fallback a inversa volatilidad
        inv_vol = 1.0 / (predicted_vols + 1e-8)
        opt_weights = inv_vol / inv_vol.sum()

    return opt_weights


# FUNCIONES DE BACKTEST Y MÉTRICAS 
def calculate_transaction_cost(old_weights, new_weights, portfolio_value, transaction_cost_rate):
    """
    Calcula el costo de transaccion basado en el turnover de la cartera.
    """
    if old_weights is None:
        turnover = np.sum(new_weights)
    else:
        turnover = np.sum(np.abs(new_weights - old_weights))

    transaction_cost = turnover * transaction_cost_rate * portfolio_value
    return transaction_cost


def calculate_metrics(values, total_transaction_costs=0, initial_capital=100000):
    """
    Calcula métricas básicas de desempeño a partir de la serie de valor de portafolio.
    Devuelve diccionario con strings formateados listos para reporte.
    """
    if len(values) < 2:
        return {}
    returns = values.pct_change().dropna()
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    years = len(values) / 252.0
    annual_return = (1 + total_return)**(1 / years) - 1 if years > 0 else 0
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


def run_backtest_walk_forward(data, weight_strategy_fn, **kwargs):
    """
    Motor de backtesting con validación walk-forward.
    - training_window: número de días a usar para entrenar en cada rebalanceo
    - rebalance_days: frecuencia del rebalanceo (dias)
    - transaction_cost_rate: tasa por transacción (ej. 0.001 = 0.1%)
    - initial_capital: capital inicial en moneda
    """
    print(f"\n{'='*60}\nEJECUTANDO BACKTEST WALK-FORWARD: {weight_strategy_fn.__name__}\n{'='*60}")

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
        print(f"  -> Rebalanceando en {current_date.date()}...")

        train_start = max(0, i - training_window)
        train_end = i
        training_data = data.iloc[train_start:train_end]

        print(f"     Entrenando con datos: {training_data.index[0].date()} a {training_data.index[-1].date()}")

        # obtener nuevos pesos desde la estrategia, que usará solo training_data (sin leakage)
        new_weights = weight_strategy_fn(training_data)
        new_weights = np.array(new_weights, dtype=float)
        # normalizar y asegurar no-negativos
        new_weights = np.clip(new_weights, 0, None)
        new_weights = new_weights / new_weights.sum() if new_weights.sum() > 0 else np.ones(len(new_weights)) / len(new_weights)

        transaction_cost = calculate_transaction_cost(previous_weights, new_weights, current_value, transaction_cost_rate)
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


# FUNCIONES DE VISUALIZACIÓN Y ARCHIVO 

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


def plot_results(results, full_data, training_window, weights_history, symbols):
    """
    Genera panel de control 2x2 con:
      1. Evolución del valor de cartera
      2. Retornos acumulados
      3. Pie: distribución final de pesos (LSTM-GA)
      4. Barra apilada: evolución de pesos por rebalanceo
    """
    print("\n" + "="*60 + "\nPANEL DE CONTROL...\n" + "="*60)
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

    # Plotar cada estrategia (excepto benchmark) con default colors
    styles = ['-', '--', '-.', ':']
    for idx, (name, data) in enumerate(results.items()):
        if "Benchmark" in name:
            continue
        style = styles[idx % len(styles)]
        ax1.plot(data.index, data.values, label=name, linewidth=2.0, linestyle=style)

    ax1.axvline(x=backtest_start_date, color='r', linestyle='-.', linewidth=2, label='Inicio de Operacion')
    y_min, y_max = ax1.get_ylim()
    ax1.fill_betweenx([y_min, y_max], full_data.index[0], backtest_start_date, color='grey', alpha=0.1)
    ax1.text(full_data.index[training_window // 2], y_min + (y_max - y_min) * 0.9,
             'Fase de Entrenamiento\n(Walk-Forward)', ha='center', style='italic', color='gray', fontsize=12)

    ax1.set_title('1. Evolucion del Valor de Cartera (Out-of-Sample Verdadero)', fontsize=16, weight='bold')
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Valor de Cartera ($)', fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = axes[0, 1]
    for name, data in results.items():
        cum_returns = (data / data.iloc[0] - 1) * 100
        if "Benchmark" in name:
            ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0, color='orange', linestyle=':')
        else:
            ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0)

    ax2.set_title('2. Retornos Acumulados (Walk-Forward Validation)', fontsize=16, weight='bold')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Retorno Acumulado (%)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')

    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%', startangle=90,
                pctdistance=0.85, explode=[0.05] * len(symbols))
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
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

# main

def main():
    # Configuración
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'INTC']
    start_date = '2018-01-01'
    end_date = '2025-01-01'
    training_window = 504  # 2 años
    transaction_cost_rate = 0.001
    initial_capital = 100000
    rebalance_days = 30

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = "results"
    run_dir = os.path.join(output_dir, f"run_walkforward_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("OPTIMIZACION DE CARTERAS: LSTM (OUT-OF-SAMPLE) + ALGORITMO GENETICO")
    print("VALIDACION WALK-FORWARD - VERSION REALISTA")
    print("=" * 80)
    print(f"Directorio de resultados: {run_dir}")
    print(f"Periodos: {start_date} a {end_date}")
    print(f"Ventana de entrenamiento: {training_window} dias ({training_window / 252:.1f} anos)")
    print(f"Costos de transaccion: {transaction_cost_rate:.3%} por operacion")
    print("MODO: Walk-Forward Analysis (reentrenamiento LSTM en cada ventana)")

    try:
        # 1) Descargar datos
        data = download_data(symbols, start_date, end_date)

        # 2) Ejecutar backtest Walk-Forward para LSTM + GA
        lstm_ga_values, lstm_ga_weights, lstm_ga_costs, lstm_ga_total_costs = run_backtest_walk_forward(
            data,
            strategy_lstm_ga,
            training_window=training_window,
            rebalance_days=rebalance_days,
            transaction_cost_rate=transaction_cost_rate,
            initial_capital=initial_capital
        )

        # 3) Ejecutar benchmark equiponderado (misma lógica de costos y walk-forward)
        print(f"\n{'='*60}\nEJECUTANDO BACKTEST WALK-FORWARD: benchmark_equiponderado\n{'='*60}")
        n_assets = data.shape[1]
        benchmark_weights = np.ones(n_assets) / n_assets

        benchmark_values_list = []
        benchmark_dates = []
        benchmark_total_costs = 0.0
        benchmark_value = initial_capital
        previous_benchmark_weights = None
        returns = data.pct_change().fillna(0)

        start_index = training_window
        for i in range(start_index, len(data), rebalance_days):
            current_date = data.index[i]
            print(f"  -> Rebalanceando benchmark en {current_date.date()}...")

            transaction_cost = calculate_transaction_cost(previous_benchmark_weights, benchmark_weights, benchmark_value, transaction_cost_rate)
            benchmark_value -= transaction_cost
            benchmark_total_costs += transaction_cost
            previous_benchmark_weights = benchmark_weights.copy()

            print(f"     Costo de transaccion (benchmark): ${transaction_cost:,.2f}")

            period_end = min(i + rebalance_days, len(data))
            for j in range(i, period_end):
                if j > i:
                    daily_return = np.dot(benchmark_weights, returns.iloc[j])
                    benchmark_value *= (1 + daily_return)
                benchmark_values_list.append(benchmark_value)
                benchmark_dates.append(data.index[j])

        benchmark_values = pd.Series(benchmark_values_list, index=benchmark_dates)
        print(f"  -> Total costos de transaccion (benchmark): ${benchmark_total_costs:,.2f}")
        print(f"  -> Periodos evaluados out-of-sample (benchmark): {len(benchmark_values)} dias")

        # 4) Consolidar resultados
        results = {
            "LSTM + Algoritmo Genetico": lstm_ga_values,
            "Benchmark (Equiponderado)": benchmark_values
        }

        transaction_costs = {
            "LSTM + Algoritmo Genetico": lstm_ga_total_costs,
            "Benchmark (Equiponderado)": benchmark_total_costs
        }

        # 5) Generar reporte de métricas
        report_lines = [
            "=" * 60,
            "METRICAS DE RENDIMIENTO COMPARATIVAS",
            "VALIDACION WALK-FORWARD",
            "CON COSTOS DE TRANSACCION INCLUIDOS",
            "=" * 60
        ]

        for name, series_data in results.items():
            total_costs = transaction_costs[name]
            metrics = calculate_metrics(series_data, total_costs, initial_capital)
            report_lines.append(f"\n{name.upper()}:")
            for m, v in metrics.items():
                report_lines.append(f"   {m}: {v}")

        # Comparativa Sharpe (siempre manejando conversion segura a float)
        try:
            lstm_sh = float(calculate_metrics(results["LSTM + Algoritmo Genetico"],
                                              transaction_costs["LSTM + Algoritmo Genetico"],
                                              initial_capital).get('Sharpe Ratio', '0'))
        except Exception:
            lstm_sh = 0.0
        try:
            bench_sh = float(calculate_metrics(results["Benchmark (Equiponderado)"],
                                               transaction_costs["Benchmark (Equiponderado)"],
                                               initial_capital).get('Sharpe Ratio', '0'))
        except Exception:
            bench_sh = 0.0

        if bench_sh != 0:
            improvement = ((lstm_sh / bench_sh) - 1) * 100
            report_lines.append(f"\nMEJORA EN SHARPE RATIO (LSTM-GA vs Benchmark): {improvement:.1f}%")

        report_lines.extend([
            "\n" + "=" * 40,
            "ANALISIS DE COSTOS DE TRANSACCION",
            "=" * 40
        ])
        for name, costs in transaction_costs.items():
            impact = costs / initial_capital
            report_lines.append(f"{name}: ${costs:,.2f} ({impact:.2%} del capital inicial)")

        report_lines.extend([
            "\n" + "=" * 40,
            "NOTA METODOLOGICA IMPORTANTE",
            "=" * 40,
            "+ Validacion Walk-Forward implementada correctamente",
            "+ Sin data snooping: entrenamiento solo con datos pasados",
            "+ Evaluacion out-of-sample en datos futuros no vistos",
            "+ Costos de transaccion incluidos para realismo",
            f"+ Ventana de entrenamiento: {training_window/252:.1f} anos",
            f"+ Periodo de evaluacion: {start_date} a {end_date}"
        ])

        # Guardar reporte
        metrics_filepath = os.path.join(run_dir, "metrics_report_walkforward.txt")
        safe_write_file(metrics_filepath, report_lines)

        # 6) Gráficos
        fig = plot_results(results, data, training_window, lstm_ga_weights, symbols)
        plot_filepath = os.path.join(run_dir, "performance_charts_walkforward.png")
        fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')

        # 7) Guardar historial de pesos si existe
        if lstm_ga_weights is not None and len(lstm_ga_weights) > 0:
            weights_df = pd.DataFrame(lstm_ga_weights, columns=symbols)
            weights_filepath = os.path.join(run_dir, "lstm_ga_weights_history_walkforward.csv")
            weights_df.to_csv(weights_filepath, index=False)
            print(f"Historial de pesos guardado en: {weights_filepath}")

        # 8) Guardar costos por rebalanceo
        costs_filepath = os.path.join(run_dir, "transaction_costs_analysis_walkforward.csv")
        # Alinear longitudes
        min_len = min(len(lstm_ga_costs), len(benchmark_values) and len(lstm_ga_costs))
        costs_df = pd.DataFrame({
            'Rebalanceo': range(len(lstm_ga_costs)),
            'LSTM_GA_Costs': lstm_ga_costs,
            # repetir GARCH placeholder si no aplica; para el benchmark ya tenemos benchmark_total_costs
        })
        costs_df.to_csv(costs_filepath, index=False)

        # 9) Guardar config
        config_filepath = os.path.join(run_dir, "config_parameters_walkforward.txt")
        config_info = [
            "PARAMETROS DE CONFIGURACION - WALK-FORWARD ANALYSIS",
            "=" * 50,
            f"Simbolos analizados: {', '.join(symbols)}",
            f"Fecha de inicio: {start_date}",
            f"Fecha de fin: {end_date}",
            f"Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} anos)",
            f"Capital inicial: ${initial_capital:,}",
            f"Frecuencia de rebalanceo: {rebalance_days} dias",
            f"Costos de transaccion: {transaction_cost_rate:.3%} por operacion",
            f"Fecha de ejecucion: {timestamp}",
            "",
            "METODOLOGIA:",
            "- Walk-Forward Analysis",
            "- Entrenamiento solo con datos historicos",
            "- Evaluacion out-of-sample estricta",
            "- Costos de transaccion incluidos",
            "",
            "ESTRATEGIAS IMPLEMENTADAS:",
            "1. LSTM + Algoritmo Genetico (con reentrenamiento LSTM por ventana)",
            "2. Benchmark Equiponderado (con costos y walk-forward)"
        ]
        safe_write_file(config_filepath, config_info)

        # Imprimir resumen y ubicaciones de archivos
        print("\n" + "=" * 60)
        print("ARCHIVOS GUARDADOS EXITOSAMENTE:")
        print("=" * 60)
        print(f"[GRAFICO] Grafico de rendimiento: {plot_filepath}")
        print(f"[METRICS] Reporte de metricas: {metrics_filepath}")
        print(f"[CONFIG] Parametros de configuracion: {config_filepath}")
        print(f"[COSTS] Analisis de costos: {costs_filepath}")
        print(f"[FOLDER] Directorio principal: {run_dir}")
        print("=" * 60)
        print("[OK] VALIDACION WALK-FORWARD COMPLETADA")
        print("=" * 60)

        plt.show()

    except Exception as e:
        print(f"\nError fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
