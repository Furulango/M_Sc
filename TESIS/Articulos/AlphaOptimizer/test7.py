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
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# Genetic algorithm (DEAP)
from deap import base, creator, tools, algorithms

# Descarga de datos

def download_data(symbols, start, end):
    all_symbols = symbols + ['^VIX']
    df = yf.download(all_symbols, start=start, end=end, auto_adjust=True)['Close']
    df = df.ffill().dropna(axis=0, how='any')
    vix_data = df[['^VIX']].copy()
    asset_data = df.drop(columns=['^VIX']).copy()
    print(f"Datos descargados: {len(asset_data.columns)} activos + VIX. Periodo: {df.index[0].date()} a {df.index[-1].date()} ({len(df)} dias).")
    return asset_data, vix_data


# -------------------------
# FEATURES (multivariadas)
# -------------------------
def create_multivariate_features(price_series, vix_series):
    """
    Crea features para predicción de retornos:
     - ret_1, ret_5, ret_21 (momentum)
     - volatility (rolling 20)
     - rsi(14), macd signal
     - vix (alineado)
    Devuelve DataFrame indexado por fechas.
    """
    p = price_series.copy()
    returns = p.pct_change()
    features = pd.DataFrame(index=p.index)

    # Momentum / returns
    features['ret_1'] = returns
    features['ret_5'] = p.pct_change(5)
    features['ret_21'] = p.pct_change(21)

    # Volatility (historic)
    features['vol_20'] = returns.rolling(window=20).std() * np.sqrt(252)

    # RSI(14)
    delta = p.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD signal
    exp1 = p.ewm(span=12, adjust=False).mean()
    exp2 = p.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    features['macd_sig'] = macd.ewm(span=9, adjust=False).mean()

    # VIX (external)
    features['vix'] = vix_series.reindex(p.index).ffill()

    features = features.dropna()
    return features


# -------------------------
# Construcción de dataset (target: future cumulative return)
# -------------------------
def create_dataset_for_return_prediction(price_series, vix_series, lookback=60, future_horizon=21):
    """
    Devuelve:
      - X (DataFrame de features a partir de fecha lookback..end-future_horizon)
      - y (array de target: future cumulative return en horizon)
    Nota: X está alineado con y en el tiempo (t -> target en t+future_horizon).
    """
    features = create_multivariate_features(price_series, vix_series)
    # target: cum. return over next future_horizon
    future_price = price_series.shift(-future_horizon)
    target = (future_price / price_series) - 1.0
    target = target.reindex(features.index)
    # Drop last rows where target is NaN
    mask = ~target.isna()
    features = features[mask]
    target = target[mask]

    # To create LSTM windows we'll need sequences later; here return raw features & target
    return features, target.values


# -------------------------
# LSTM model helpers
# -------------------------
def build_lstm_model(input_shape, units1=64, units2=32, dropout=0.25, lr=0.001):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units2),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def create_lstm_dataset_from_features(features_df, target_array, lookback=60):
    """
    Construye X,y para LSTM a partir de features DataFrame y target array (alineados por índice).
    X shape: (n_samples, lookback, n_features)
    """
    X, y = [], []
    data_np = features_df.values
    for i in range(lookback, len(data_np)):
        X.append(data_np[i - lookback:i, :])
        y.append(target_array[i])
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y).reshape(-1,)


# -------------------------
# Predict function: ensemble LSTM + RF
# -------------------------
def train_and_predict_ensemble(price_series, vix_series, model_bundle, lookback=60, future_horizon=21,
                               lstm_epochs=30, rf_estimators=100, verbose=False):
    """
    model_bundle: dict per symbol with keys: {'lstm': model or None, 'rf': model or None, 'scaler': scaler}
    Retorna: predicted_future_return (float), updated_model_bundle
    Strategy:
      - Prepara features & target usando ventana disponible.
      - Entrena/finetunea LSTM (en escalado) y RandomForest (en estándar).
      - Predice el return promedio del horizon usando ensemble weighted average.
    """
    try:
        features, target = create_dataset_for_return_prediction(price_series, vix_series, lookback=lookback, future_horizon=future_horizon)
        if len(features) < lookback + 10:
            # Poca data -> fallback median historical return
            hist_ret = price_series.pct_change(future_horizon).dropna()
            if len(hist_ret) == 0:
                return 0.0, model_bundle
            return float(hist_ret.mean()), model_bundle

        # Split last segment as 'train' (we always train on the full available training window)
        # For practical purposes, train on all data except final lookback that we will use to predict
        scaler = model_bundle.get('scaler')
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features.values)
        features_scaled = pd.DataFrame(scaler.transform(features.values), index=features.index, columns=features.columns)

        # Create LSTM sequences
        X_lstm, y_lstm = create_lstm_dataset_from_features(features_scaled, target, lookback=lookback)
        # For RF we can use tabular features (e.g., the last row of each window)
        X_rf = []
        y_rf = []
        for i in range(lookback, len(features)):
            X_rf.append(features.values[i - lookback:i, :].mean(axis=0))  # aggregated window features
            y_rf.append(target[i])
        X_rf = np.array(X_rf)
        y_rf = np.array(y_rf)

        # SAFEGUARD
        if len(X_lstm) == 0 or len(X_rf) == 0:
            hist_ret = price_series.pct_change(future_horizon).dropna()
            return float(hist_ret.mean() if len(hist_ret) > 0 else 0.0), model_bundle

        # Train or finetune LSTM
        lstm_model = model_bundle.get('lstm')
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        if lstm_model is None:
            if verbose: print("      -> Creando LSTM...")
            lstm_model = build_lstm_model(input_shape=(lookback, features.shape[1]), units1=64, units2=32, dropout=0.25, lr=0.001)
            lstm_model.fit(X_lstm, y_lstm, epochs=lstm_epochs, batch_size=32, verbose=0, callbacks=[early_stop])
        else:
            if verbose: print("      -> Fine-tuning LSTM...")
            tf.keras.backend.set_value(lstm_model.optimizer.learning_rate, 0.0005)
            lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0, callbacks=[early_stop])

        # Train or update RandomForest
        rf_model = model_bundle.get('rf')
        if rf_model is None:
            if verbose: print("      -> Training RandomForest...")
            rf_model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
            rf_model.fit(X_rf, y_rf)
        else:
            # re-fit by combining previous data with new (simple: re-fit on current training data)
            if verbose: print("      -> Re-fitting RandomForest...")
            rf_model.fit(X_rf, y_rf)

        # Last available sequence for prediction (most recent lookback window)
        last_features = features_scaled.values[-lookback:]
        last_seq = last_features.reshape(1, lookback, features.shape[1])

        pred_lstm = float(lstm_model.predict(last_seq, verbose=0)[0, 0])
        # For RF, use mean/aggregate of last window
        last_rf_input = features.values[-lookback:].mean(axis=0).reshape(1, -1)
        pred_rf = float(rf_model.predict(last_rf_input)[0])

        # Ensemble: weighted average (tunable)
        ensemble_pred = 0.6 * pred_lstm + 0.4 * pred_rf

        # Save updated bundle
        model_bundle.update({'lstm': lstm_model, 'rf': rf_model, 'scaler': scaler})
        return float(ensemble_pred), model_bundle

    except Exception as e:
        print(f"      -> ERROR ensemble prediction: {e}. Fallback to historical mean.")
        hist_ret = price_series.pct_change(future_horizon).dropna()
        return float(hist_ret.mean() if len(hist_ret) > 0 else 0.0), model_bundle


# -------------------------
# ALGORITMO GENÉTICO (fitness basado en expected sharpe)
# -------------------------
def genetic_algorithm_optimizer(expected_returns, cov_matrix, previous_weights=None,
                                population_size=120, generations=80, cxpb=0.6, mutpb=0.25):
    n_assets = len(expected_returns)
    # Crear tipos si no existen
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    max_weight_constraint = 0.5  # permitir mayor concentración cuando la señal lo indique
    turnover_penalty_factor = 0.75  # penalizar más el turnover

    def evaluate(individual):
        w = np.array(individual, dtype=float)
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        if np.any(w > max_weight_constraint):
            # penaliza fuertemente individuos que violen max weight
            return (-9999,)

        port_return = np.dot(w, expected_returns)  # expected return based on model predictions
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) + 1e-9
        # Reward risk-adjusted return; subtract constant risk-free (2%)
        sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else -9999

        # Penalizar turnover si previous_weights disponible
        turnover = 0.0
        if previous_weights is not None:
            turnover = np.sum(np.abs(w - previous_weights))
        fitness = sharpe - (turnover * turnover_penalty_factor)
        return (float(fitness),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=float)
    best = np.clip(best, 0, None)
    if best.sum() == 0:
        best = np.ones_like(best) / len(best)
    else:
        best = best / best.sum()
    return best, hof[0].fitness.values[0]


# -------------------------
# Estrategia híbrida LSTM-RF + GA (usa retornos predichos)
# -------------------------
def strategy_lstm_rf_ga(historical_data, vix_data, previous_weights=None, model_bundles=None,
                        lookback=60, future_horizon=21, verbose=False):
    """
    Para cada activo:
     - crea features sobre la ventana histórica provista
     - entrena/finetunea ensemble LSTM+RF y predice el return futuro
    Luego:
     - calcula matriz de covarianza sobre returns históricos
     - corre GA para optimizar Sharpe esperado usando los retornos predichos
    """
    returns = historical_data.pct_change().dropna()
    if returns.empty:
        n = historical_data.shape[1]
        return np.ones(n) / n

    expected_returns = []
    # Mantener orden consistente con columnas
    symbols = historical_data.columns.tolist()
    for symbol in symbols:
        if verbose: print(f"    -> Predicting for {symbol}...")
        price_series = historical_data[symbol]
        model_bundle = model_bundles.get(symbol, {})
        pred_ret, updated_bundle = train_and_predict_ensemble(price_series, vix_data[symbol] if symbol in vix_data.columns else vix_data['^VIX'],
                                                              model_bundle, lookback=lookback, future_horizon=future_horizon, verbose=verbose)
        model_bundles[symbol] = updated_bundle
        expected_returns.append(pred_ret)
    expected_returns = np.array(expected_returns, dtype=float)

    # If predictions are all nan or zero, fallback
    if np.all(np.isnan(expected_returns)) or np.all(np.isclose(expected_returns, 0.0)):
        inv_vol = 1.0 / (returns.std().values + 1e-9)
        weights = inv_vol / inv_vol.sum()
        return weights

    cov_matrix = returns.cov().values * 252.0

    try:
        opt_weights, _ = genetic_algorithm_optimizer(expected_returns, cov_matrix, previous_weights=previous_weights,
                                                     population_size=120, generations=80)
    except Exception as e:
        # fallback to inverse volatility
        inv_vol = 1.0 / (returns.std().values + 1e-9)
        opt_weights = inv_vol / inv_vol.sum()

    return opt_weights


# -------------------------
# Backtest Walk-Forward con threshold de rebalanceo
# -------------------------
def calculate_transaction_cost(old_weights, new_weights, portfolio_value, transaction_cost_rate):
    if old_weights is None:
        turnover = np.sum(new_weights)
    else:
        turnover = np.sum(np.abs(new_weights - old_weights))
    return turnover * transaction_cost_rate * portfolio_value


def calculate_metrics(values, total_transaction_costs=0, initial_capital=100000):
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
    cost_impact = total_transaction_costs / initial_capital if initial_capital > 0 else 0
    metrics = {
        'Retorno Anual': f"{annual_return:.2%}", 'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}", 'Max Drawdown': f"{drawdown.min():.2%}",
        'Valor Final': f"${values.iloc[-1]:,.0f}"
    }
    if total_transaction_costs > 0:
        metrics['Costos Transaccion'] = f"${total_transaction_costs:,.2f}"
        metrics['Impacto en Rendimiento'] = f"{cost_impact:.2%}"
    return metrics


def run_backtest_walk_forward(data, vix_data, weight_strategy_fn, model_bundles, **kwargs):
    print(f"\n{'='*60}\nEJECUTANDO BACKTEST WALK-FORWARD MEJORADO: {weight_strategy_fn.__name__}\n{'='*60}")

    initial_capital = kwargs.get('initial_capital', 100000)
    training_window = kwargs.get('training_window', 1008)  # mantengo 4 años por defecto
    rebalance_days = kwargs.get('rebalance_days', 126)    # semestral aprox (126 dias)
    transaction_cost_rate = kwargs.get('transaction_cost_rate', 0.001)
    rebalance_threshold = kwargs.get('rebalance_threshold', 0.10)  # 10% L1 change mínimo para rebalancear

    print(f"  -> Ventana de entrenamiento: {training_window} dias ({training_window/252:.1f} años)")
    print(f"  -> Frecuencia de rebalanceo: {rebalance_days} dias (umbral {rebalance_threshold*100:.1f}% para cambios)")
    print(f"  -> Transaction cost rate: {transaction_cost_rate*100:.2f}% por turnover")

    portfolio_values = []
    weights_history = []
    dates = []
    returns = data.pct_change().fillna(0)
    current_value = initial_capital
    previous_weights = None
    total_transaction_costs = 0
    start_index = training_window

    for i in range(start_index, len(data), rebalance_days):
        current_date = data.index[i]
        print(f"\n  -> Ventana hasta {current_date.date()} (rebalanceando)...")
        train_start = max(0, i - training_window)
        train_end = i
        training_data_assets = data.iloc[train_start:train_end]
        training_data_vix = vix_data.reindex(training_data_assets.index).ffill()

        # Obtener nuevas propuestas de pesos
        new_weights = weight_strategy_fn(training_data_assets, training_data_vix,
                                         previous_weights=previous_weights, model_bundles=model_bundles)

        new_weights = np.clip(np.array(new_weights, dtype=float), 0, None)
        if new_weights.sum() == 0:
            new_weights = np.ones_like(new_weights) / len(new_weights)
        else:
            new_weights = new_weights / new_weights.sum()

        # Aplicar umbral: solo rebalancear si la suma absoluta de cambios > threshold
        if previous_weights is not None:
            change = np.sum(np.abs(new_weights - previous_weights))
            if change < rebalance_threshold:
                # skip rebalancing; mantener previous_weights
                applied_weights = previous_weights.copy()
                print(f"    -> Cambio {change:.4f} < umbral {rebalance_threshold:.4f}. No rebalancea.")
            else:
                applied_weights = new_weights.copy()
                transaction_cost = calculate_transaction_cost(previous_weights, applied_weights, current_value, transaction_cost_rate)
                current_value -= transaction_cost
                total_transaction_costs += transaction_cost
                print(f"    -> Rebalanceo aplicado. Costo de transaccion: ${transaction_cost:,.2f}")
        else:
            # primer rebalanceo: aplicar new_weights y cobrar un coste inicial proporcional
            applied_weights = new_weights.copy()
            transaction_cost = calculate_transaction_cost(None, applied_weights, current_value, transaction_cost_rate)
            current_value -= transaction_cost
            total_transaction_costs += transaction_cost
            print(f"    -> Primer rebalanceo. Costo de transaccion: ${transaction_cost:,.2f}")

        weights_history.append(applied_weights)
        previous_weights = applied_weights.copy()

        period_end = min(i + rebalance_days, len(data))
        for j in range(i, period_end):
            if j > i:
                daily_return = np.dot(applied_weights, returns.iloc[j])
                current_value *= (1 + daily_return)
            portfolio_values.append(current_value)
            dates.append(data.index[j])

    print(f"\n  -> Total costos de transaccion: ${total_transaction_costs:,.2f}")
    return (pd.Series(portfolio_values, index=dates),
            np.array(weights_history),
            None,
            total_transaction_costs)


# -------------------------
# Gráficas e informes (mantengo la estructura original)
# -------------------------
def safe_write_file(filepath, content):
    if isinstance(content, list):
        content = '\n'.join(content)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except UnicodeEncodeError:
        with open(filepath, 'w', encoding='ascii', errors='ignore') as f:
            f.write(content)


def plot_results(results, full_data, training_window, weights_history, symbols):
    print("\n" + "="*60 + "\nPANEL DE CONTROL...\n" + "="*60)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    ax1 = axes[0, 0]
    initial_capital = 100000
    backtest_start_date = full_data.index[training_window]

    n_assets = full_data.shape[1]
    benchmark_returns = (full_data.pct_change().fillna(0).loc[backtest_start_date:] @ (np.ones(n_assets) / n_assets)) + 1
    benchmark_values = initial_capital * benchmark_returns.cumprod()

    ax1.plot(benchmark_values.index, benchmark_values.values,
             label='Benchmark (Equiponderado)', linewidth=1.5, color='orange', linestyle=':')

    for name, data in results.items():
        if "Benchmark" in name: continue
        ax1.plot(data.index, data.values, label=name, linewidth=2.0)

    ax1.axvline(x=backtest_start_date, color='r', linestyle='-.', linewidth=2, label='Inicio de Operacion')
    ax1.set_title('1. Evolucion del Valor de Cartera', fontsize=16, weight='bold')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax2 = axes[0, 1]
    for name, data in results.items():
        cum_returns = (data / data.iloc[0] - 1) * 100
        if "Benchmark" in name:
             ax2.plot(benchmark_values.index, (benchmark_values / benchmark_values.iloc[0] - 1) * 100, label=name, linewidth=2.0, color='orange', linestyle=':')
        else:
            ax2.plot(cum_returns.index, cum_returns.values, label=name, linewidth=2.0)

    ax2.set_title('2. Retornos Acumulados', fontsize=16, weight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--')

    ax3 = axes[1, 0]
    if len(weights_history) > 0:
        final_weights = weights_history[-1]
        ax3.pie(final_weights, labels=symbols, autopct='%1.1f%%', startangle=90)
    ax3.set_title('3. Distribucion Final de Pesos (LSTM-RF-GA)', fontsize=16, weight='bold')

    ax4 = axes[1, 1]
    if len(weights_history) > 1:
        weights_df = pd.DataFrame(weights_history, columns=symbols)
        weights_df.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis', legend=None)

    ax4.set_title('4. Evolucion de Pesos en Rebalanceos (LSTM-RF-GA)', fontsize=16, weight='bold')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout(pad=3.0)
    fig.suptitle('Panel de Control - LSTM + RF Ensemble', fontsize=22, weight='bold')
    plt.subplots_adjust(top=0.92)
    return fig


# -------------------------
# MAIN
# -------------------------
def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'INTC']
    start_date = '2010-01-01'
    end_date = '2025-01-01'

    # Parametros
    training_window = 1008  # dias
    rebalance_days = 126    # semestral
    transaction_cost_rate = 0.001
    initial_capital = 100000
    lookback = 60
    future_horizon = 21

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = "results"
    run_dir = os.path.join(output_dir, f"run_ensemble_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("="*80)
    print("OPTIMIZACION DE CARTERAS: LSTM-RF Ensemble + GA (MEJORADO)")
    print("VALIDACION WALK-FORWARD - ENSEMBLE + FITNESS SHARPE")
    print("="*80)

    try:
        asset_data, vix_data = download_data(symbols, start_date, end_date)
        # Preparar dict para guardar modelos por símbolo
        model_bundles = {s: {} for s in symbols}

        lstm_rf_values, lstm_rf_weights, _, lstm_rf_total_costs = run_backtest_walk_forward(
            asset_data,
            vix_data,
            strategy_lstm_rf_ga,
            model_bundles,
            training_window=training_window,
            rebalance_days=rebalance_days,
            transaction_cost_rate=transaction_cost_rate,
            initial_capital=initial_capital,
        )

        # Recalcular benchmark para periodo operativo
        backtest_start_date = asset_data.index[training_window]
        n_assets = asset_data.shape[1]
        benchmark_returns = (asset_data.pct_change().fillna(0).loc[backtest_start_date:] @ (np.ones(n_assets) / n_assets)) + 1
        benchmark_values = initial_capital * benchmark_returns.cumprod()
        benchmark_total_costs = calculate_transaction_cost(None, np.ones(n_assets)/n_assets, initial_capital, transaction_cost_rate)

        results_for_metrics = {"LSTM-RF Ensemble + GA": lstm_rf_values, "Benchmark (Equiponderado)": benchmark_values}
        transaction_costs = {"LSTM-RF Ensemble + GA": lstm_rf_total_costs, "Benchmark (Equiponderado)": benchmark_total_costs}

        # Metrics report
        report_lines = ["=" * 60, "METRICAS DE RENDIMIENTO (ENSEMBLE LSTM-RF)", "=" * 60]
        for name, series_data in results_for_metrics.items():
            total_costs = transaction_costs[name]
            metrics = calculate_metrics(series_data, total_costs, initial_capital)
            report_lines.append(f"\n{name.upper()}:")
            for m, v in metrics.items():
                report_lines.append(f"   {m}: {v}")

        metrics_filepath = os.path.join(run_dir, "metrics_report_ensemble.txt")
        safe_write_file(metrics_filepath, report_lines)

        results = {"LSTM-RF Ensemble + GA": lstm_rf_values, "Benchmark (Equiponderado)": benchmark_values}
        fig = plot_results(results_for_metrics, asset_data, training_window, lstm_rf_weights, symbols)
        plot_filepath = os.path.join(run_dir, "performance_charts_ensemble.png")
        fig.savefig(plot_filepath, dpi=300, bbox_inches='tight')

        print("\nReporte guardado en:", metrics_filepath)
        print("Grafica guardada en:", plot_filepath)
        print("\n" + "="*60 + "\n[OK] VALIDACION COMPLETADA\n" + "="*60)
        plt.show()

    except Exception as e:
        print(f"\nError fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
