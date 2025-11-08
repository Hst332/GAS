# ----------------------------------------------------------
# 🔥 Erdgas Tagesprognose (TradingEconomics + Automatisiert)
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # deinen API-Key einfügen!
SYMBOL_GAS = "natural gas"
SYMBOL_OIL = "crude oil"

SMA_SHORT, SMA_LONG = 15, 40
W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL = 8, 1.0, 5, 1.5, 8
CHAIN_MAX = 14
ATR_PERIOD = 14
RSI_PERIOD = 14

# ----------------------------------------------------------
# 📥 Daten laden (letzte 90 Tage)
# ----------------------------------------------------------
def load_data_te():
    print("⏳ Lade aktuelle Daten von TradingEconomics ...")
    url_gas = f"https://api.tradingeconomics.com/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
    url_oil = f"https://api.tradingeconomics.com/commodity/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"

    r_gas = requests.get(url_gas)
    r_oil = requests.get(url_oil)
    if r_gas.status_code != 200 or r_oil.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

    gas = pd.DataFrame(r_gas.json())
    oil = pd.DataFrame(r_oil.json())

    gas["Date"] = pd.to_datetime(gas["Date"])
    oil["Date"] = pd.to_datetime(oil["Date"])
    gas = gas.rename(columns={"Value": "Close"})
    oil = oil.rename(columns={"Value": "Oil_Close_prev"})

    df = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Oil_Change"] = df["Oil_Close_prev"].pct_change().fillna(0)
    return df.tail(120).reset_index(drop=True)

df = load_data_te()

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    df["ATR"] = (df["Close"].rolling(ATR_PERIOD).std() * 2).fillna(method="bfill")
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi().fillna(method="bfill")
    return df

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Trendberechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    prob = 50
    prob += w_sma if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -w_sma
    prob += (df["RSI"].iloc[-1]-50) * w_rsi / 10
    atr_last = df["ATR"].iloc[-1]
    if atr_last > 0:
        prob += np.tanh((df["Return"].iloc[-1]/atr_last)*2)*w_atr
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r)==sign)
    prob += sign * streak * w_streak
    prob += df["Oil_Change"].iloc[-1] * 100 * w_oil
    return max(0, min(100, prob))

# ----------------------------------------------------------
# 🔍 Tagesprognose berechnen
# ----------------------------------------------------------
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "📈 Steigend" if trend_prob >= 50 else "📉 Fallend"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 📊 Ergebnis
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas (Natural Gas): {round(last_close, 3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob, 2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob, 2)} %\n"
    f"⚙️ Parameter → SMA={SMA_SHORT}/{SMA_LONG}, W_SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, Oil={W_OIL}"
)

print(msg)

# Ausgabe speichern (optional)
with open("forecast_result.txt", "w") as f:
    f.write(msg)
