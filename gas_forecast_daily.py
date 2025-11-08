# ----------------------------------------------------------
# 🔥 Erdgas Tagesprognose (TradingEconomics API, 4 Jahre)
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_API_KEY_HIER"   # <<-- Hier deinen TradingEconomics-API-Key einfügen

SYMBOL_GAS = "NGAS"
SYMBOL_OIL = "OIL"

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14

# Beste Parameter aus Optimierung
SMA_SHORT = 15
SMA_LONG = 40
W_SMA = 8
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.5
W_OIL = 8

# ----------------------------------------------------------
# 📥 Datenabruf von TradingEconomics
# ----------------------------------------------------------
def load_data_te():
    print("⏳ Lade aktuelle Daten von TradingEconomics ...")

    base = "https://api.tradingeconomics.com/historical/commodity"
    key = TRADINGECONOMICS_KEY

    url_gas = f"{base}/{SYMBOL_GAS}?c={key}"
    url_oil = f"{base}/{SYMBOL_OIL}?c={key}"

    r_gas = requests.get(url_gas)
    r_oil = requests.get(url_oil)

    if r_gas.status_code != 200 or r_oil.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

    gas = pd.DataFrame(r_gas.json())
    oil = pd.DataFrame(r_oil.json())

    gas.rename(columns=str.title, inplace=True)
    oil.rename(columns=str.title, inplace=True)

    gas["Date"] = pd.to_datetime(gas["Date"])
    oil["Date"] = pd.to_datetime(oil["Date"])
    oil["Oil_Close_prev"] = oil["Close"].shift(1)

    df = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Oil_Change"] = df["Oil_Close_prev"].pct_change().fillna(0)

    df = df.tail(4 * 365).reset_index(drop=True)
    return df

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    return df.bfill()

# ----------------------------------------------------------
# 🔮 Trend-Berechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    prob = 50
    prob += w_sma if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -w_sma
    prob += (df["RSI"].iloc[-1] - 50) * w_rsi / 10

    atr_last = df["ATR"].iloc[-1]
    if atr_last > 0:
        prob += np.tanh((df["Return"].iloc[-1] / atr_last) * 2) * w_atr

    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * w_streak
    prob += df["Oil_Change"].iloc[-1] * 100 * w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 📈 Hilfsfunktionen
# ----------------------------------------------------------
def get_streak(df):
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    direction = "steigend 📈" if recent_returns[-1] > 0 else "fallend 📉"
    streak_len = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == np.sign(recent_returns[-1]))
    return direction, streak_len

# ----------------------------------------------------------
# 🚀 Hauptprogramm
# ----------------------------------------------------------
df = load_data_te()
df = add_indicators(df)

trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

streak_direction, streak_length = get_streak(df)

# ----------------------------------------------------------
# 🧾 Ausgabe
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas (NGAS): {round(last_close, 3)} USD/MMBtu\n"
    f"🔮 Trendprognose: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob, 2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob, 2)} %\n"
    f"📈 Aktueller Trend: {streak_length} Tage in Folge {streak_direction}\n"
    f"⚙️ Parameter → SMA={SMA_SHORT}/{SMA_LONG}, WSMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, OIL={W_OIL}\n"
)

print(msg)
with open("daily_forecast.txt", "w") as f:
    f.write(msg)

print("✅ Tagesprognose erfolgreich gespeichert (daily_forecast.txt)")
