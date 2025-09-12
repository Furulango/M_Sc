import os
import warnings
import random
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor

# Genetic algorithm (DEAP)
from deap import base, creator, tools, algorithms

# Silenciar advertencias comunes para una salida más limpia
warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel('ERROR')


# -----------------------------------------------------------------------------
# 1. DESCARGA Y PREPARACIÓN DE DATOS
# -----------------------------------------------------------------------------

def download_data(symbols, start, end):
    """Descarga datos de precios (Close) y Volumen para activos, y VIX."""
    all_symbols = symbols + ['^VIX']
    # Descargamos todos los datos necesarios (Close y Volume)
    df_full = yf.download(all_symbols, start=start, end=end, auto_adjust=True)
    
    # Rellenar datos faltantes y eliminar filas con NaNs
    df_full = df_full.ffill().dropna(axis=0, how='any')
    
    # Separar los datos
    asset_prices = df_full['Close'][symbols].copy()
    asset_volumes = df_full['Volume'][symbols].copy()
    vix_data = df_full['Close'][['^VIX']].copy()
    
    print(f"Datos descargados: {len(asset_prices.columns)} activos + VIX.")
    print(f"Periodo: {df_full.index[0].date()} a {df_full.index[-1].date()} ({len(df_full)} dias).")
    
    return asset_prices, asset_volumes, vix_data


# -----------------------------------------------------------------------------
# 2. INGENIERÍA DE FEATURES (MEJORADA)
# -----------------------------------------------------------------------------

# --- Funciones auxiliares para Features Mejorados ---

def calculate_bollinger_position(price_series, window=20, num_std=2):
    """Calcula la posición del precio dentro de las Bandas de Bollinger (0 a 1)."""
    rolling_mean = price_series.rolling(window).mean()
    rolling_std = price_series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    position = (price_series - lower_band) / (upper_band - lower_band + 1e-9)
    return position.clip(0, 1) # Aseguramos que el valor esté entre 0 y 1

def calculate_volume_ratio(volume_series, window=20):
    """Compara el volumen actual con su media móvil."""
    rolling_avg_volume = volume_series.rolling(window).mean()
    return volume_series / (rolling_avg_volume + 1e-9)

def _hurst(ts):
    """Calcula el exponente de Hurst para una serie temporal."""
    lags = range(2, 100)
    # Evitar divisiones por cero o std de cero
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 1e-9]
    if not tau: return 0.5
    lags = range(2, len(tau) + 2)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def calculate_hurst_exponent(returns, window=100):
    """Aplica el cálculo de Hurst de forma rolling."""
    return returns.rolling(window).apply(_hurst, raw=True)

def _entropy(series, bins=10):
    """Calcula la entropía de Shannon para una serie."""
    counts = np.histogram(series, bins=bins)[0]
    return shannon_entropy(counts / (len(series) + 1e-9))

def calculate_entropy(returns, window=20):
    """Aplica el cálculo de entropía de forma rolling."""
    return returns.rolling(window).apply(_entropy, raw=True)


def create_enhanced_features(price_series, volume_series, vix_series):
    """
    Crea un DataFrame de features multivariadas y mejoradas para una serie de precios.
    """
    features = pd.DataFrame(index=price_series.index)
    returns = price_series.pct_change()
    
    # 1. Features de Momentum y Volatilidad (mejorados)
    for period in [1, 5, 10, 21, 63]:
        features[f'ret_{period}'] = price_series.pct_change(period)
        features[f'vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
    # 2. Indicadores técnicos adicionales
    features['bb_position'] = calculate_bollinger_position(price_series)
    features['volume_ratio'] = calculate_volume_ratio(volume_series)
    
    # 3. Features de microestructura (pueden ser lentos en ventanas grandes)
    features['hurst'] = calculate_hurst_exponent(returns)
    features['entropy'] = calculate_entropy(returns)
        
    # 4. Régimen de mercado
    features['vix_percentile'] = vix_series.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # 5. Feature externa: VIX
    features['vix'] = vix_series.reindex(price_series.index).ffill()

    return features.dropna()


# -----------------------------------------------------------------------------
# 3. CONSTRUCCIÓN DE DATASET Y MODELO LSTM (MEJORADO)
# -----------------------------------------------------------------------------

def create_dataset_for_return_prediction(price_series, volume_series, vix_series, lookback=60, future_horizon=21):
    """Prepara los datos de features y target para la predicción de retornos."""
    features = create_enhanced_features(price_series, volume_series, vix_series)
    
    # Target: retorno acumulado en el horizonte futuro
    future_price = price_series.shift(-future_horizon)
    target = (future_price / price_series) - 1.0
    
    # Alineamos features y target
    target = target.reindex(features.index)
    mask = ~target.isna()
    features = features[mask]
    target = target[mask]
    
    return features, target.values

def create_lstm_dataset_with_proper_scaling(features_df, target_array, lookback=60, train_size=0.8):
    """
    SOLUCIÓN A DATA LEAKAGE: Escala los datos y crea secuencias LSTM.
    El scaler se ajusta SOLAMENTE con los datos de entrenamiento para evitar look-ahead bias.
    """
    # 1. Dividir el índice para ajustar el scaler solo en datos de entrenamiento
    split_idx = int(len(features_df) * train_size)
    
    # 2. Ajustar el scaler SOLO en la porción de entrenamiento
    scaler = StandardScaler()
    scaler.fit(features_df.iloc[:split_idx].values)
    
    # 3. Aplicar la transformación a todo el dataset
    features_scaled = scaler.transform(features_df.values)
    
    # 4. Crear las secuencias (X, y) para el LSTM
    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback:i, :])
        y.append(target_array[i])
        
    if not X:
        return np.array([]), np.array([]), scaler
        
    return np.array(X), np.array(y).reshape(-1,), scaler


def build_improved_lstm_model(input_shape, units1=64, units2=32, dropout=0.3, lr=0.001):
    """
    SOLUCIÓN A OVERFITTING: Modelo LSTM más robusto con regularización y clipping.
    """
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(dropout),
        
        LSTM(units2, return_sequences=True, 
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(dropout),
        
        LSTM(units2 // 2),
        Dropout(dropout),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=lr, clipnorm=1.0) # Gradient clipping
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model


# -----------------------------------------------------------------------------
# 4. PREDICCIÓN ENSEMBLE (LSTM + RandomForest)
# -----------------------------------------------------------------------------

def train_and_predict_ensemble(price_series, volume_series, vix_series, model_bundle, lookback=60, future_horizon=21,
                               lstm_epochs=30, rf_estimators=100, verbose=False):
    """Entrena y predice usando un ensemble de LSTM y RandomForest."""
    try:
        features, target = create_dataset_for_return_prediction(price_series, volume_series, vix_series, 
                                                                lookback=lookback, future_horizon=future_horizon)
        if len(features) < lookback + 10:
            hist_ret = price_series.pct_change(future_horizon).dropna()
            return float(hist_ret.mean()) if not hist_ret.empty else 0.0, model_bundle

        # Crear dataset LSTM con escalado correcto
        X_lstm, y_lstm, scaler = create_lstm_dataset_with_proper_scaling(features, target, lookback)

        # Para RF, usamos features tabulares (media de la ventana)
        X_rf = np.array([features.values[i-lookback:i].mean(axis=0) for i in range(lookback, len(features))])
        y_rf = target[lookback:]

        if X_lstm.size == 0 or X_rf.size == 0:
            hist_ret = price_series.pct_change(future_horizon).dropna()
            return float(hist_ret.mean()) if not hist_ret.empty else 0.0, model_bundle

        # Entrenar o re-entrenar LSTM
        lstm_model = model_bundle.get('lstm')
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        if lstm_model is None:
            if verbose: print("      -> Creando LSTM...")
            lstm_model = build_improved_lstm_model(input_shape=(lookback, features.shape[1]))
            lstm_model.fit(X_lstm, y_lstm, epochs=lstm_epochs, batch_size=32, verbose=0, callbacks=[early_stop])
        else: # Fine-tuning
            if verbose: print("      -> Fine-tuning LSTM...")
            tf.keras.backend.set_value(lstm_model.optimizer.learning_rate, 0.0005)
            lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0, callbacks=[early_stop])
            
        # Entrenar o re-entrenar RandomForest
        rf_model = model_bundle.get('rf')
        if rf_model is None:
            if verbose: print("      -> Creando RandomForest...")
            rf_model = RandomForestRegressor(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
        rf_model.fit(X_rf, y_rf)

        # Predicción con la última secuencia de datos
        last_features_scaled = scaler.transform(features.values[-lookback:])
        last_seq_lstm = last_features_scaled.reshape(1, lookback, features.shape[1])
        pred_lstm = float(lstm_model.predict(last_seq_lstm, verbose=0)[0, 0])

        last_seq_rf = features.values[-lookback:].mean(axis=0).reshape(1, -1)
        pred_rf = float(rf_model.predict(last_seq_rf)[0])

        ensemble_pred = 0.6 * pred_lstm + 0.4 * pred_rf
        model_bundle.update({'lstm': lstm_model, 'rf': rf_model, 'scaler': scaler})
        return float(ensemble_pred), model_bundle

    except Exception as e:
        print(f"      -> ERROR en predicción ensemble: {e}. Usando media histórica.")
        hist_ret = price_series.pct_change(future_horizon).dropna()
        return float(hist_ret.mean()) if not hist_ret.empty else 0.0, model_bundle


# -----------------------------------------------------------------------------
# 5. ALGORITMO GENÉTICO (MEJORADO)
# -----------------------------------------------------------------------------

def enhanced_genetic_optimizer(expected_returns, cov_matrix, previous_weights=None,
                               population_size=120, generations=80, cxpb=0.6, mutpb=0.25,
                               max_position=0.25, risk_free=0.02):
    """
    SOLUCIÓN A GA SIMPLE: Optimizador genético con función de fitness mejorada.
    Incluye CVaR, penalización por concentración (Herfindahl) y por turnover.
    """
    n_assets = len(expected_returns)

    # Crear tipos de DEAP si no existen
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Inicialización inteligente
    def init_smart_individual():
        if previous_weights is not None:
            base_weights = previous_weights + np.random.normal(0, 0.05, n_assets)
        else: # Risk parity como punto de partida
            inv_vol = 1 / (np.diag(cov_matrix) ** 0.5 + 1e-9)
            base_weights = inv_vol / inv_vol.sum()
            base_weights += np.random.normal(0, 0.05, n_assets)
        return creator.Individual(np.maximum(base_weights, 0).tolist())

    toolbox.register("individual", init_smart_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_portfolio(individual):
        w = np.array(individual, dtype=float)
        w = np.maximum(w, 0)
        w_sum = w.sum()
        w = w / (w_sum + 1e-10)

        # 1. Restricción dura de peso máximo
        if np.any(w > max_position):
            return (-1e10,)

        # 2. Métricas de riesgo-retorno
        port_return = np.dot(w, expected_returns)
        port_vol = np.sqrt(np.dot(w.T, cov_matrix @ w))

        # 3. CVaR (Conditional Value at Risk)
        returns_sim = np.random.multivariate_normal(expected_returns, cov_matrix, 1000)
        portfolio_returns = returns_sim @ w
        cvar_95 = np.mean(portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 5)])

        # 4. Sharpe y penalizaciones
        sharpe = (port_return - risk_free) / (port_vol + 1e-9)
        herfindahl = np.sum(w**2) # Penalización por concentración

        turnover_penalty = 0
        if previous_weights is not None:
            turnover = np.sum(np.abs(w - previous_weights))
            turnover_penalty = turnover * 1.5 # Factor de penalización
        
        # 5. Función de fitness compuesta
        fitness = (sharpe * 1.0 + cvar_95 * 0.5 - herfindahl * 0.5 - turnover_penalty)
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
# 6. ESTRATEGIA Y BACKTESTING (CON REBALANCEO ADAPTATIVO)
# -----------------------------------------------------------------------------

def strategy_lstm_rf_ga(historical_data, volume_data, vix_data, previous_weights=None, model_bundles=None,
                        lookback=60, future_horizon=21, verbose=False):
    """
    Orquesta la predicción con el ensemble y la optimización con el GA mejorado.
    """
    returns = historical_data.pct_change().dropna()
    if returns.empty:
        return np.ones(historical_data.shape[1]) / historical_data.shape[1]

    expected_returns = []
    symbols = historical_data.columns.tolist()

    for symbol in symbols:
        if verbose: print(f"    -> Prediciendo para {symbol}...")
        price_series = historical_data[symbol]
        volume_series = volume_data[symbol]
        vix_series = vix_data['^VIX']
        model_bundle = model_bundles.get(symbol, {})
        
        pred_ret, updated_bundle = train_and_predict_ensemble(
            price_series, volume_series, vix_series, model_bundle,
            lookback=lookback, future_horizon=future_horizon, verbose=verbose
        )
        model_bundles[symbol] = updated_bundle
        expected_returns.append(pred_ret)
        
    expected_returns = np.array(expected_returns, dtype=float)
    
    # Annualizar retornos predichos para el GA
    expected_returns_annualized = (1.0 + expected_returns) ** (252.0 / future_horizon) - 1.0
    cov_matrix = returns.cov().values * 252.0

    if verbose:
        print("    -> Retornos anualizados esperados:", np.round(expected_returns_annualized, 3))

    try:
        opt_weights, fitness = enhanced_genetic_optimizer(
            expected_returns_annualized, cov_matrix, previous_weights=previous_weights
        )
        if verbose:
            print(f"    -> GA finalizado. Fitness: {fitness:.4f}")
            print("    -> Pesos optimizados:", np.round(opt_weights, 3))
    except Exception as e:
        if verbose: print(f"    -> ERROR en GA: {e}. Usando inverse-volatility.")
        inv_vol = 1.0 / (returns.std().values + 1e-9)
        opt_weights = inv_vol / inv_vol.sum()

    return opt_weights


def determine_market_regime(vix_series, data):
    """Determina el régimen de mercado basado en volatilidad y tendencia."""
    # Régimen de volatilidad (percentil del VIX en el último año)
    vol_percentile = vix_series.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).iloc[-1]
    
    # Fuerza de la tendencia (cruce de medias móviles en el índice equiponderado)
    market_index = data.mean(axis=1)
    short_ma = market_index.rolling(50).mean().iloc[-1]
    long_ma = market_index.rolling(200).mean().iloc[-1]
    trend_strength = abs(short_ma - long_ma) / market_index.iloc[-1]
    
    return {'volatility': vol_percentile, 'trend': trend_strength}


def adaptive_rebalancing(current_weights, proposed_weights, market_regime):
    """
    SOLUCIÓN A REBALANCEO FIJO: Decide si rebalancear y cómo, basado en el régimen de mercado.
    """
    volatility_regime = market_regime['volatility'] # 0 a 1
    trend_strength = market_regime['trend'] # > 0
    
    # Ajustar umbral dinámicamente
    base_threshold = 0.05
    if volatility_regime > 0.8: # Alta volatilidad -> menos rebalanceos
        threshold = base_threshold * 1.5
    elif trend_strength > 0.05: # Tendencia fuerte -> más rebalanceos
        threshold = base_threshold * 0.7
    else: # Mercado normal
        threshold = base_threshold
        
    weight_change = np.sum(np.abs(proposed_weights - current_weights))
    
    if weight_change > threshold:
        # Suavizado adaptativo: más agresivo si el cambio propuesto es grande
        alpha = min(0.8, weight_change * 2) 
        new_weights = alpha * proposed_weights + (1 - alpha) * current_weights
        new_weights /= new_weights.sum()
        return new_weights, True, f"Cambio ({weight_change:.2%}) > Umbral ({threshold:.2%})"
        
    return current_weights, False, f"Cambio ({weight_change:.2%}) <= Umbral ({threshold:.2%})"


def calculate_transaction_cost(old_weights, new_weights, portfolio_value, cost_rate=0.001):
    turnover = np.sum(np.abs(new_weights - old_weights)) if old_weights is not None else np.sum(new_weights)
    return turnover * cost_rate * portfolio_value


def run_backtest_walk_forward(data, volume_data, vix_data, weight_strategy_fn, model_bundles, **kwargs):
    print(f"\n{'='*60}\nEJECUTANDO BACKTEST WALK-FORWARD MEJORADO\n{'='*60}")

    # Configuración del backtest
    initial_capital, training_window, rebalance_days, cost_rate = 100000, 1008, 63, 0.001
    
    portfolio_values, dates, weights_history = [], [], []
    current_value = initial_capital
    previous_weights, total_costs = None, 0
    returns = data.pct_change().fillna(0)

    for i in range(training_window, len(data), rebalance_days):
        current_date = data.index[i]
        print(f"\n  -> Rebalanceando en {current_date.date()}...")
        
        # 1. Definir ventana de entrenamiento
        train_start, train_end = max(0, i - training_window), i
        training_data_assets = data.iloc[train_start:train_end]
        training_data_volumes = volume_data.iloc[train_start:train_end]
        training_data_vix = vix_data.reindex(training_data_assets.index).ffill()

        # 2. Obtener pesos propuestos de la estrategia
        proposed_weights = weight_strategy_fn(
            training_data_assets, training_data_volumes, training_data_vix,
            previous_weights=previous_weights, model_bundles=model_bundles, verbose=True
        )

        # 3. Aplicar Rebalanceo Adaptativo
        if previous_weights is not None:
            market_regime = determine_market_regime(training_data_vix['^VIX'], training_data_assets)
            applied_weights, rebalanced, reason = adaptive_rebalancing(previous_weights, proposed_weights, market_regime)
            print(f"    -> Decisión adaptativa: {'Rebalancear' if rebalanced else 'Mantener'}. Razón: {reason}")
        else: # Primer rebalanceo
            applied_weights, rebalanced = proposed_weights, True
            print("    -> Primer rebalanceo: aplicando pesos iniciales.")
        
        # 4. Calcular costos si se rebalanceó
        if rebalanced:
            cost = calculate_transaction_cost(previous_weights, applied_weights, current_value, cost_rate)
            current_value -= cost
            total_costs += cost
            print(f"    -> Costo de transacción: ${cost:,.2f}")
        
        weights_history.append(applied_weights)
        previous_weights = applied_weights.copy()

        # 5. Simular el período hasta el próximo rebalanceo
        period_end = min(i + rebalance_days, len(data))
        for j in range(i, period_end):
            daily_return = np.dot(applied_weights, returns.iloc[j])
            current_value *= (1 + daily_return)
            portfolio_values.append(current_value)
            dates.append(data.index[j])
            
    print(f"\n  -> Total costos de transacción: ${total_costs:,.2f}")
    return pd.Series(portfolio_values, index=dates), np.array(weights_history), total_costs


# -----------------------------------------------------------------------------
# 7. VISUALIZACIÓN Y REPORTES
# -----------------------------------------------------------------------------

def calculate_metrics(values, total_costs=0, initial_capital=100000):
    """Calcula métricas de rendimiento clave."""
    if len(values) < 2: return {}
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    annual_return = (1 + total_return)**(1 / years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / (annual_vol + 1e-9)
    drawdown = (values / values.cummax() - 1).min()
    cost_impact = total_costs / initial_capital
    return {
        'Retorno Anual': f"{annual_return:.2%}", 'Volatilidad Anual': f"{annual_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}", 'Max Drawdown': f"{drawdown:.2%}",
        'Valor Final': f"${values.iloc[-1]:,.0f}", 'Costos de Transacción': f"${total_costs:,.2f}",
        'Impacto de Costos': f"{cost_impact:.2%}"
    }

def plot_results(results_series, weights_history, symbols, training_window, data):
    """Genera un panel de control con los resultados del backtest."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 14), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Panel de Control - Estrategia LSTM-RF Ensemble + GA Mejorado', fontsize=22, weight='bold')

    # 1. Evolución del valor de la cartera
    ax1 = axes[0, 0]
    for name, series in results_series.items():
        ax1.plot(series.index, series.values, label=name, linewidth=2.5 if "Ensemble" in name else 1.5)
    ax1.set_title('1. Evolución del Valor de Cartera', fontsize=16)
    ax1.legend()
    ax1.grid(True, linestyle=':')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Drawdown
    ax2 = axes[0, 1]
    for name, series in results_series.items():
        drawdown = (series / series.cummax() - 1) * 100
        ax2.plot(drawdown.index, drawdown.values, label=name, linewidth=2 if "Ensemble" in name else 1)
    ax2.set_title('2. Drawdown de la Cartera (%)', fontsize=16)
    ax2.legend()
    ax2.grid(True, linestyle=':')
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.1)

    # 3. Distribución final de pesos
    ax3 = axes[1, 0]
    final_weights = weights_history[-1]
    non_zero_weights = final_weights[final_weights > 0.005]
    non_zero_symbols = np.array(symbols)[final_weights > 0.005]
    ax3.pie(non_zero_weights, labels=non_zero_symbols, autopct='%1.1f%%', startangle=90)
    ax3.set_title('3. Distribución Final de Pesos', fontsize=16)

    # 4. Evolución de pesos
    ax4 = axes[1, 1]
    weights_df = pd.DataFrame(weights_history, columns=symbols)
    weights_df.plot(kind='area', stacked=True, ax=ax4, colormap='viridis', legend=None)
    ax4.set_title('4. Evolución de Pesos en Rebalanceos', fontsize=16)
    ax4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# -----------------------------------------------------------------------------
# 8. EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    # --- Configuración ---
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH']
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    initial_capital = 100000
    training_window = 1008  # ~4 años
    
    try:
        # Descargar datos
        asset_data, volume_data, vix_data = download_data(symbols, start_date, end_date)
        
        # Inicializar contenedores de modelos
        model_bundles = {s: {} for s in symbols}

        # Ejecutar backtest
        ensemble_values, ensemble_weights, ensemble_costs = run_backtest_walk_forward(
            asset_data, volume_data, vix_data, strategy_lstm_rf_ga, model_bundles,
            training_window=training_window
        )
        
        # Crear Benchmark Equiponderado para el mismo período
        backtest_start_date = ensemble_values.index[0]
        n_assets = asset_data.shape[1]
        benchmark_returns = (asset_data.pct_change().fillna(0).loc[backtest_start_date:] @ (np.ones(n_assets) / n_assets)) + 1
        benchmark_values = initial_capital * benchmark_returns.cumprod()

        # Generar reporte
        print("\n" + "="*60 + "\nREPORTE FINAL DE MÉTRICAS\n" + "="*60)
        ensemble_metrics = calculate_metrics(ensemble_values, ensemble_costs, initial_capital)
        benchmark_metrics = calculate_metrics(benchmark_values, 0, initial_capital)
        
        print("\nESTRATEGIA ENSEMBLE LSTM-RF + GA MEJORADO:")
        for m, v in ensemble_metrics.items(): print(f"  {m}: {v}")
            
        print("\nBENCHMARK (EQUIPONDERADO):")
        for m, v in benchmark_metrics.items(): print(f"  {m}: {v}")

        # Generar y mostrar gráfico
        results = {"Estrategia Ensemble": ensemble_values, "Benchmark (1/N)": benchmark_values}
        fig = plot_results(results, ensemble_weights, symbols, training_window, asset_data)
        plt.show()

    except Exception as e:
        print(f"\nError fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
