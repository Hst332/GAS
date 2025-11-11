# ----------------------------------------------------------
# 🔹 Daily Natural Gas Forecast mit Speicherung der Historie
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import ta
import re
import os

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
HIST_FILE = "gas_history.csv"
SYMBOL_GAS = "Natural Gas"
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"

# Modellparameter (optimiertes Modell)
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
# 🔹 Historische Daten laden
# ----------------------------------------------------------
try:
    df = pd.read_csv(HIST_FILE, parse_dates=["Date"])
    print(f"✅ Historische Daten geladen: {len(df)} Tage")
except FileNotFoundError:
    print("⚠️ Keine historische Datei gefunden. Neue wird erstellt.")
    df = pd.DataFrame(columns=["Date", "Close", "High", "Low"])

# ----------------------------------------------------------
# 🔹 Aktuellen Kurs von Finanzen.net holen
# ----------------------------------------------------------
try:
    url = "https://www.finanzen.net/rohstoffe/erdgas-preis-natural-gas"
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    match = re.search(r'([0-9]+,[0-9]+)\s*USD', html)
    if match:
        today_price = float(match.group(1).replace(',', '.'))
        today = pd.Timestamp(datetime.now().date())
        if not ((df["Date"] == today).any()):
            new_row = pd.DataFrame([{"Date": today, "Close": today_price, "High": today_price, "Low": today_price}])
            df = pd.concat([df, new_row], ignore_index=True)
        print(f"✅ Aktueller Preis von Finanzen.net: {today_price} USD")
    else:
        raise ValueError("❌ Kurs konnte nicht gefunden werden.")
except Exception as e:
    raise ValueError(f"❌ Fehler beim Abrufen des aktuellen Preises: {e}")

# ----------------------------------------------------------
# 🔹 Indikatoren berechnen
# ----------------------------------------------------------
df["High"] = df.get("High", df["Close"])
df["Low"] = df.get("Low", df["Close"])
df["Return"] = df["Close"].pct_change().fillna(0)

# ATR
high, low, close = df["High"], df["Low"], df["Close"]
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

# RSI
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi().bfill()

# SMA
df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

# ----------------------------------------------------------
# 🔹 Wahrscheinlichkeit berechnen
# ----------------------------------------------------------
def calculate_prediction(df):
    prob = 50
    prob += W_SMA if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -W_SMA
    prob += (df["RSI"].iloc[-1] - 50) * W_RSI / 10
    prob += np.tanh(df["Return"].iloc[-1] / df["ATR"].iloc[-1]) * W_ATR
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * W_STREAK
    return max(0, min(100, prob))

trend_prob = calculate_prediction(df)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 🔹 Ergebnis ausgeben & speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgaspreis: {round(last_close,3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"⚙️ Modellparameter → SMA={SMA_SHORT}/{SMA_LONG}, W_SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}"
)

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("✅ Ergebnis in result.txt gespeichert.")

# ----------------------------------------------------------
# 🔹 Historie speichern für morgen
# ----------------------------------------------------------
df.to_csv(HIST_FILE, index=False)
print(f"💾 Historische Daten in {HIST_FILE} gespeichert.")

# ----------------------------------------------------------
# 🔹 Änderungserkennung (>10 %) + GitHub-Notification
# ----------------------------------------------------------
PREVIOUS_FILE = "previous_result.txt"

def get_previous_value(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        m = re.search(r'Wahrscheinlichkeit steigend:\s*([0-9.]+)', text)
        if m:
            return float(m.group(1))
        return None

previous_value = get_previous_value(PREVIOUS_FILE)
current_value = trend_prob

if previous_value is not None:
    change = abs(current_value - previous_value) / previous_value * 100 if previous_value != 0 else 0
    print(f"🔸 Änderung gegenüber letzter Prognose: {change:.2f}%")
    if change > 10:
        print("::warning::⚠️ Änderung >10 % erkannt!")
else:
    print("ℹ️ Kein Vergleichswert vorhanden (erster Lauf).")

# Aktuellen Wert speichern
with open(PREVIOUS_FILE, "w", encoding="utf-8") as f:
    f.write(msg)
print("💾 previous_result.txt aktualisiert.")
