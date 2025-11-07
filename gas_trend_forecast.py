# ----------------------------------------------------------
# 🔥 Erdgas Trend Forecast (robust)
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
SYMBOL = "NG=F"
OIL_SYMBOL = "CL=F"

ATR_PERIOD = 14
RSI_PERIOD = 14
SMA_SHORT = 15
SMA_LONG = 40
CHAIN_MAX = 14

W_SMA = 5
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.0
OIL_WEIGHT = 5

END = datetime.now()
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data(ticker, oil_ticker):
    gas = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    oil = yf.download(oil_ticker, start=START, end=END, auto_adjust=True, progress=False)

    gas = gas.rename(columns=str.title).reset_index()
    gas["Return"] = gas["Close"].pct_change().fillna(0)
    gas["Date"] = pd.to_datetime(gas["Date"])

    oil = oil.rename(columns=str.title).reset_index()
    oil["Oil_Close_prev"] = oil["Close"].shift(1)

    gas = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    gas["Oil_Change"] = gas["Oil_Close_prev"].pct_change().fillna(0)
    return gas

df = load_data(SYMBOL, OIL_SYMBOL)
print(f"✅ Loaded {len(df)} days of Natural Gas data.")

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    for col in ["High", "Low", "Close"]:
        series = df.get(col)
        if series is None or series.empty or series.isnull().all():
            df[col] = df["Close"]

    # ATR berechnen (robust)
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()

    return df.bfill()

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Prognoseberechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    last = df.iloc[-1]
    prob = 50

    # SMA Trend
    prob += w_sma if last.sma_short > last.sma_long else -w_sma

    # RSI
    prob += (last.RSI - 50) * w_rsi / 10

    # ATR-weighted move
    if last.ATR > 0:
        daily_move = df["Return"].iloc[-1]
        prob += np.tanh((daily_move / last.ATR) * 2) * w_atr

    # Trendserie
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    streak = 0
    sign = np.sign(recent_returns[-1])
    for r in reversed(recent_returns[:-1]):
        if np.sign(r) == sign:
            streak += 1
        else:
            break
    prob += sign * streak * w_streak

    # Ölpreis Einfluss
    prob += df["Oil_Change"].iloc[-1] * 100 * w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 🔹 Aktueller Trend
# ----------------------------------------------------------
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, OIL_WEIGHT, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 🔹 Trendserie
# ----------------------------------------------------------
def get_streak(df):
    recent_returns = df["Return"].tail(30).values
    up = recent_returns[-1] > 0
    streak = 1
    for r in reversed(recent_returns[:-1]):
        if (r > 0 and up) or (r < 0 and not up):
            streak += 1
        else:
            break
    direction = "gestiegen 📈" if up else "gefallen 📉"
    return direction, streak

streak_direction, streak_length = get_streak(df)

# ----------------------------------------------------------
# 📊 Ergebnis ausgeben & speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 Erdgas: {round(last_close,2)} $\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"📏 Aktueller Trend: Erdgas ist {streak_length} Tage in Folge {streak_direction}\n"
)

print(msg)

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert ✅")
