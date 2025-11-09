# ----------------------------------------------------------
# 🔥 Erdgas Tagesforecast mit TradingEconomics (3 Versuche + Wochenend-Fallback)
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
import time
import os

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # <– Deinen API-Key hier einsetzen

SYMBOL_GAS = "Natural Gas"
SYMBOL_OIL = "Crude Oil"

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14

# Optimierte Parameter aus vorherigem Lauf
SMA_SHORT, SMA_LONG = 15, 40
W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL = 8, 1.0, 5, 1.5, 8

# ----------------------------------------------------------
# 📥 Daten laden mit 3 Versuchen
# ----------------------------------------------------------
def load_data_te(max_retries=3, delay=5):
    for attempt in range(1, max_retries + 1):
        print(f"⏳ Lade Daten von TradingEconomics (Versuch {attempt}) ...")
        url_gas = f"https://api.tradingeconomics.com/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
        url_oil = f"https://api.tradingeconomics.com/commodity/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"

        r_gas = requests.get(url_gas)
        r_oil = requests.get(url_oil)

        if r_gas.status_code == 200 and r_oil.status_code == 200:
            gas = pd.DataFrame(r_gas.json())
            oil = pd.DataFrame(r_oil.json())

            gas["Date"] = pd.to_datetime(gas["Date"])
            oil["Date"] = pd.to_datetime(oil["Date"])
            oil["Oil_Close_prev"] = oil["Close"].shift(1)

            df = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
            df["Return"] = df["Close"].pct_change().fillna(0)
            df["Oil_Change"] = df["Oil_Close_prev"].pct_change().fillna(0)
            return df

        print(f"⚠️ Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")
        if attempt < max_retries:
            print(f"⏳ Warte {delay} Sekunden und versuche erneut...")
            time.sleep(delay)

    # Wenn alle Versuche fehlschlagen
    raise ValueError(f"Fehler nach {max_retries} Versuchen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    df["ATR"] = (
        pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
    ).rolling(ATR_PERIOD).mean().bfill()

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    return df.bfill()

# ----------------------------------------------------------
# 🔮 Vorhersageberechnung
# ----------------------------------------------------------
def calculate_prediction(df):
    if len(df) < SMA_LONG:
        return 50

    df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

    prob = 50
    prob += W_SMA if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -W_SMA
    prob += (df["RSI"].iloc[-1] - 50) * W_RSI / 10

    atr_last = df["ATR"].iloc[-1]
    if atr_last > 0:
        prob += np.tanh((df["Return"].iloc[-1] / atr_last) * 2) * W_ATR

    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * W_STREAK
    prob += df["Oil_Change"].iloc[-1] * 100 * W_OIL

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 🚀 Hauptablauf mit Fallback
# ----------------------------------------------------------
try:
    # Prüfen, ob Wochenende ist
    weekday = datetime.now().weekday()  # 0=Mo ... 6=So
    if weekday >= 5:
        raise RuntimeError("Wochenende")

    df = load_data_te()
    df = add_indicators(df)
    prob = calculate_prediction(df)
    trend = "Steigend 📈" if prob >= 50 else "Fallend 📉"
    last_close = df["Close"].iloc[-1]

    msg = (
        f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
        f"🔥 Erdgas Preis: {round(last_close, 3)} USD/MMBtu\n"
        f"🔮 Trend: {trend}\n"
        f"📊 Wahrscheinlichkeit steigend: {round(prob, 2)} %\n"
        f"📊 Wahrscheinlichkeit fallend : {round(100 - prob, 2)} %"
    )

except RuntimeError as e:
    msg = (
        f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
        f"⚠️ Heute ist kein Handelstag ({e}). Keine aktuellen Daten verfügbar.\n"
        f"📁 Letzter bekannter Trend bleibt bestehen."
    )
except Exception as e:
    msg = (
        f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
        f"⚠️ Warnung: Keine aktuellen Daten verfügbar, Fehler: {e}"
    )

# ----------------------------------------------------------
# 💾 Ergebnis speichern
# ----------------------------------------------------------
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)

print(msg)
print("📁 Ergebnis gespeichert in result.txt ✅")
