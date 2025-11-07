# ----------------------------------------------------------
# 🧠 Natural Gas Parameter Optimizer
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Settings
# ----------------------------------------------------------
SYMBOL = "NG=F"    # Natural Gas
OIL_SYMBOL = "CL=F"  # WTI Crude Oil
START = datetime.now() - timedelta(days=3*365)
END = datetime.now()

# Parameter-Ranges für Optimierung
SMA_SHORT_RANGE = [10, 15, 20]
SMA_LONG_RANGE = [30, 40, 50]
W_SMA_RANGE = [4, 5, 6]
W_RSI_RANGE = [0.8, 1.0, 1.2]
W_ATR_RANGE = [4, 5, 6]
W_STREAK_RANGE = [0.5, 1.0, 1.5]
OIL_WEIGHT_RANGE = [0, 5, 10]

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14

# ----------------------------------------------------------
# 📥 Load data
# ----------------------------------------------------------
def load_data():
    gas = yf.download(SYMBOL, start=START, end=END, auto_adjust=True, progress=False)
    oil = yf.download(OIL_SYMBOL, start=START, end=END, auto_adjust=True, progress=False)

    gas = gas.rename(columns=str.title).reset_index()
    gas["Return"] = gas["Close"].pct_change().fillna(0)
    gas["Date"] = pd.to_datetime(gas["Date"])

    oil = oil.rename(columns=str.title).reset_index()
    oil["Oil_Close_prev"] = oil["Close"].shift(1)
    gas = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").fillna(method="ffill")
    gas["Oil_Change"] = gas["Oil_Close_prev"].pct_change().fillna(0)
    return gas

df = load_data()
print(f"✅ Loaded {len(df)} days of Natural Gas data.")

# ----------------------------------------------------------
# 📊 Indicators
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=ATR_PERIOD).average_true_range()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    return df.bfill()

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Forecast model
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    last = df.iloc[-1]
    prob = 50

    # Trend SMA
    prob += w_sma if last.sma_short > last.sma_long else -w_sma

    # RSI
    prob += (last.RSI - 50) * w_rsi / 10

    # ATR-weighted move
    if last.ATR > 0:
        daily_move = df["Return"].iloc[-1]
        prob += np.tanh((daily_move / last.ATR) * 2) * w_atr

    # Streak
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    streak = 0
    sign = np.sign(recent_returns[-1])
    for r in reversed(recent_returns[:-1]):
        if np.sign(r) == sign:
            streak += 1
        else:
            break
    prob += sign * streak * w_streak

    # Oil effect
    prob += df["Oil_Change"].iloc[-1] * 100 * w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 🎯 Backtest
# ----------------------------------------------------------
def backtest(df, params):
    acc = 0
    count = 0
    for i in range(60, len(df)):
        sub = df.iloc[:i].copy()
        p = calculate_prediction(sub, **params)
        predicted_up = p >= 50
        actual_up = df["Return"].iloc[i] > 0
        if predicted_up == actual_up:
            acc += 1
        count += 1
    return acc / count * 100 if count else 0

# ----------------------------------------------------------
# 🧪 Parameter optimization
# ----------------------------------------------------------
results = []

for s_short in SMA_SHORT_RANGE:
    for s_long in SMA_LONG_RANGE:
        if s_long <= s_short:
            continue
        for w_sma in W_SMA_RANGE:
            for w_rsi in W_RSI_RANGE:
                for w_atr in W_ATR_RANGE:
                    for w_streak in W_STREAK_RANGE:
                        for w_oil in OIL_WEIGHT_RANGE:
                            params = dict(
                                w_sma=w_sma,
                                w_rsi=w_rsi,
                                w_atr=w_atr,
                                w_streak=w_streak,
                                w_oil=w_oil,
                                sma_short=s_short,
                                sma_long=s_long,
                            )
                            acc = backtest(df, params)
                            results.append((acc, params))
                            print(f"→ {acc:.2f}%  {params}")

best = max(results, key=lambda x: x[0])
print("\n🏁 Beste Kombination:")
print(best[1])
print(f"🎯 Trefferquote: {best[0]:.2f} %")
