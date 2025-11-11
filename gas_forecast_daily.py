# ----------------------------------------------------------
# 🔥 Daily Natural Gas Forecast mit TradingEconomics + Fallback
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import time
import re

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # <--- Hier deinen API-Key einfügen!
SYMBOL_GAS = "NG"   # ✅ offizielles Kürzel für Natural Gas Futures
SYMBOL_OIL = "CL"   # ✅ Crude Oil WTI Futures

# Modellparameter (deine optimierten Werte)
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

# ----------------------------------------------------------
# 📥 Datenabruf mit Wiederholungen und Fallback
# ----------------------------------------------------------
def load_data():
    for attempt in range(3):
        print(f"⏳ Lade Daten von TradingEconomics (Versuch {attempt+1}) ...")
        try:
            # Verwende neue, stabile API-Endpunkte
            url_gas = f"https://api.tradingeconomics.com/commodity/historical/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
            url_oil = f"https://api.tradingeconomics.com/commodity/historical/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"
            r_gas = requests.get(url_gas, timeout=10)
            r_oil = requests.get(url_oil, timeout=10)

            if r_gas.status_code == 200 and r_oil.status_code == 200:
                gas = pd.DataFrame(r_gas.json())
                oil = pd.DataFrame(r_oil.json())

                gas["Date"] = pd.to_datetime(gas["date"])
                gas["Close"] = pd.to_numeric(gas["value"], errors="coerce")
                gas = gas.dropna(subset=["Close"])
                gas = gas.sort_values("Date")

                print(f"✅ {len(gas)} Gasdatenpunkte geladen.")
                return gas
            else:
                print(f"⚠️ Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")
        except Exception as e:
            print(f"⚠️ Fehler: {e}")
        if attempt < 2:
            print("⏳ Warte 5 Sekunden und versuche erneut...")
            time.sleep(5)

    # Fallback zu Finanzen.net
    print("⚠️ TE-Daten nicht verfügbar – hole aktuellen Kurs von Finanzen.net ...")
    try:
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
    except Exception as e:
        print(f"❌ Fehler beim Fallback: {e}")
    raise ValueError("❌ Keine Daten verfügbar (TE und Finanzen.net fehlgeschlagen).")

# ----------------------------------------------------------
# 📈 Berechnungen
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    if "High" not in df.columns:
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
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
# 🧠 Hauptlogik
# ----------------------------------------------------------
try:
    df = load_data()
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
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    print("✅ Ergebnis in result.txt gespeichert.")

except Exception as e:
    err = f"❌ Fehler: {e}"
    print(err)
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(err)
