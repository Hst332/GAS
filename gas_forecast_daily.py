# ----------------------------------------------------------
# 🔥 Erdgas Tagesprognose (TradingEconomics, 4 Jahre Historie)
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # <-- Deinen API-Key hier einsetzen!

SYMBOL_GAS = "natural-gas"   # ✅ Korrekte Symbole für TradingEconomics
SYMBOL_OIL = "crude-oil"

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14
ROLL_WINDOW = 30

# Optimierte Parameter aus letztem Forecast
SMA_SHORT, SMA_LONG = 15, 40
W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL = 8, 1.0, 5, 1.5, 8

# Zeitraum: letzte 4 Jahre
END = datetime.now()
START = END - timedelta(days=4*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data_te():
    print("⏳ Lade aktuelle Daten von TradingEconomics ...")
    url_gas = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
    url_oil = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"

    r_gas = requests.get(url_gas)
    r_oil = requests.get(url_oil)

    if r_gas.status_code != 200 or r_oil.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

    gas = pd.DataFrame(r_gas.json())
    oil = pd.DataFrame(r_oil.json())

    # Struktur angleichen
    gas = gas.rename(columns=str.title)
    oil = oil.rename(columns=str.title)
    gas["Date"] = pd.to_datetime(gas["Date"])
    oil["Date"] = pd.to_datetime(oil["Date"])

    # Berechnungen
    gas["Return"] = gas["Close"].pct_change().fillna(0)
    oil["Oil_Close_prev"] = oil["Close"].shift(1)
    gas = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    gas["Oil_Change"] = gas["Oil_Close_prev"].pct_change().fillna(0)

    # Nur letzte 4 Jahre
    gas = gas[gas["Date"] >= START].reset_index(drop=True)
    return gas

df = load_data_te()
print(f"✅ Loaded {len(df)} days of data ({df['Date'].iloc[0]} → {df['Date'].iloc[-1]})")

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    high, low, close = df["Close"], df["Close"], df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi().bfill()
    return df

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Trendberechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    df = df.copy()
    if len(df) < sma_long:
        return 50

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
# 📈 Berechnung für heute
# ----------------------------------------------------------
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "📈 Steigend" if trend_prob >= 50 else "📉 Fallend"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 📊 Ergebnis anzeigen
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas (Natural Gas): {round(last_close, 3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob, 2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob, 2)} %\n"
    f"⚙️ Parameter → SMA={SMA_SHORT}/{SMA_LONG}, WSMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, OIL={W_OIL}\n"
)

print(msg)

# ----------------------------------------------------------
# 💾 Ausgabe speichern
# ----------------------------------------------------------
with open("daily_forecast.txt", "w") as f:
    f.write(msg)

print("✅ Tagesprognose gespeichert: daily_forecast.txt")
