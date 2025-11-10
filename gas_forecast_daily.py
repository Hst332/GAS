# ----------------------------------------------------------
# 🔥 Daily Natural Gas Forecast mit Fallback zu Finanzen.net
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import time
import re
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # <--- Hier deinen API-Key einsetzen
SYMBOL_GAS = "Natural Gas"
SYMBOL_OIL = "Crude Oil"

# ----------------------------------------------------------
# 🔧 Modellparameter (von deinem optimierten Modell)
# ----------------------------------------------------------
SMA_SHORT = 15
SMA_LONG = 40
W_SMA = 8
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.5
W_OIL = 8
ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14
ROLL_WINDOW = 30

# ----------------------------------------------------------
# 📥 Daten abrufen mit Fallback (TradingEconomics → Finanzen.net)
# ----------------------------------------------------------
def load_data_te_or_finanzen():
    for attempt in range(3):
        print(f"⏳ Lade Daten von TradingEconomics (Versuch {attempt+1}) ...")
        try:
            url_gas = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
            r_gas = requests.get(url_gas, timeout=10)
            if r_gas.status_code == 200:
                gas = pd.DataFrame(r_gas.json())
                gas["Date"] = pd.to_datetime(gas["Date"])
                gas["Close"] = pd.to_numeric(gas["Close"], errors="coerce")
                gas = gas.dropna(subset=["Close"])
                print(f"✅ {len(gas)} historische Datenpunkte geladen.")
                return gas
            else:
                print(f"⚠️ Fehler beim Abrufen: Gas {r_gas.status_code}")
        except Exception as e:
            print(f"⚠️ Fehler: {e}")
        if attempt < 2:
            print("⏳ Warte 5 Sekunden und versuche erneut...")
            time.sleep(5)

    # Fallback zu Finanzen.net
    print("⚠️ TE-Daten nicht verfügbar – hole aktuellen Kurs von Finanzen.net ...")
    url = "https://www.finanzen.net/rohstoffe/erdgas-preis-natural-gas"
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    match = re.search(r'([0-9]+,[0-9]+)\s*USD', html)
    if match:
        price = float(match.group(1).replace(',', '.'))
        df = pd.DataFrame([{
            "Date": datetime.now(),
            "Close": price,
            "High": price,
            "Low": price
        }])
        print(f"✅ Aktueller Gaspreis von Finanzen.net: {price} USD")
        return df
    else:
        raise ValueError("❌ Weder TE noch Finanzen.net liefern Daten.")

# ----------------------------------------------------------
# 📈 Indikatoren & Berechnung
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    if "High" not in df.columns:
        df["High"] = df["Close"]
        df["Low"] = df["Close"]

    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    df["Return"] = df["Close"].pct_change().fillna(0)
    return df.bfill()

def calculate_prediction(df):
    df = add_indicators(df)
    if len(df) < SMA_LONG:
        return 50
    df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

    prob = 50
    prob += W_SMA if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -W_SMA
    prob += (df["RSI"].iloc[-1]-50) * W_RSI / 10
    prob += np.tanh(df["Return"].iloc[-1] / df["ATR"].iloc[-1]) * W_ATR

    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * W_STREAK
    return max(0, min(100, prob))

# ----------------------------------------------------------
# 📊 Hauptablauf (mit Datenherkunft)
# ----------------------------------------------------------
try:
    df = load_data_te_or_finanzen()

    # Herkunft bestimmen
    source = "TradingEconomics" if len(df) > 1 else "Finanzen.net"
    last_update = df["Date"].iloc[-1].strftime("%d.%m.%Y %H:%M")

    trend_prob = calculate_prediction(df)
    trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
    last_close = df["Close"].iloc[-1]

    msg = (
        f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
        f"🔥 Erdgaspreis: {round(last_close,3)} USD/MMBtu\n"
        f"🔮 Trend: {trend}\n"
        f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
        f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
        f"⚙️ Modellparameter → SMA={SMA_SHORT}/{SMA_LONG}, W_SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}\n"
        f"🕒 Datenquelle: {source} (Letztes Update: {last_update})"
    )

    print(msg)

    # Ergebnis speichern
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    print("✅ Ergebnis in result.txt gespeichert.")

except Exception as e:
    print(f"❌ Fehler: {e}")
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(f"❌ Fehler: {e}")

