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
import sys
import subprocess

# ----------------------------------------------------------
# ✅ BeautifulSoup sicher importieren
# ----------------------------------------------------------
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("⚠️ bs4 nicht gefunden. Installiere...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4"])
    from bs4 import BeautifulSoup
print("✅ BeautifulSoup bereit.")

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
HIST_FILE = "gas_history.csv"
PREVIOUS_FILE = "previous_result.txt"
SMA_SHORT = 15
SMA_LONG = 40
W_SMA = 8
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.5
ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14
API_KEY = os.getenv("TRADINGECONOMICS_KEY", "DEIN_KEY_HIER")

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
# 🔹 Aktuellen Spotpreis holen (TradingEconomics → Fallback finanzen.net)
# ----------------------------------------------------------
today_price = None
try:
    # 1️⃣ Versuch: TradingEconomics API
    url = f"https://api.tradingeconomics.com/markets/commodities?c={API_KEY}"
    data = requests.get(url, timeout=10).json()

    for item in data:
        if item.get("symbol") == "NATGAS" or "Natural Gas" in item.get("name", ""):
            today_price = float(item.get("last", 0))
            break

    if today_price and 1 < today_price < 50:
        print(f"✅ Spotpreis (TradingEconomics): {today_price} USD/MMBtu")
    else:
        raise ValueError("Kein gültiger Wert von TradingEconomics erhalten")

except Exception as e:
    print(f"⚠️ TradingEconomics-API nicht verfügbar ({e}) – wechsle zu finanzen.net ...")
    try:
        # 2️⃣ Fallback: finanzen.net Parsing
        url = "https://www.finanzen.net/rohstoffe/erdgas-preis-natural-gas"
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        soup = BeautifulSoup(html, "html.parser")

        candidates = []
        for tag in soup.find_all(text=re.compile(r"([0-9]+,[0-9]+)\s*USD")):
            if "MMBtu" in tag or "Erdgas" in tag or "Natural Gas" in tag:
                val = re.search(r"([0-9]+,[0-9]+)", tag)
                if val:
                    price = float(val.group(1).replace(",", "."))
                    if 1 < price < 50:
                        candidates.append(price)

        if not candidates:
            for el in soup.find_all(["span", "div"], class_=re.compile(r"price", re.I)):
                m = re.search(r"([0-9]+,[0-9]+)", el.text)
                if m:
                    price = float(m.group(1).replace(",", "."))
                    if 1 < price < 50:
                        candidates.append(price)

        if candidates:
            today_price = min(candidates)
            print(f"✅ Spotpreis (Fallback finanzen.net): {today_price} USD/MMBtu")
        else:
            raise ValueError("❌ Kein Spotpreis auf finanzen.net gefunden!")

    except Exception as e2:
        print(f"⚠️ Fehler beim Abrufen des Preises: {e2}")
        if not df.empty:
            today_price = df["Close"].iloc[-1]
            print(f"ℹ️ Verwende letzten bekannten Preis: {today_price} USD/MMBtu")
        else:
            raise SystemExit("❌ Kein Preis verfügbar und keine Historie vorhanden — Abbruch.")

# ----------------------------------------------------------
# 🔹 Neue Zeile speichern
# ----------------------------------------------------------
today = pd.Timestamp(datetime.now().date())
if not ((df["Date"] == today).any()):
    new_row = pd.DataFrame([{"Date": today, "Close": today_price, "High": today_price, "Low": today_price}])
    df = pd.concat([df, new_row], ignore_index=True)

# ----------------------------------------------------------
# 🔹 Indikatoren berechnen
# ----------------------------------------------------------
df["High"] = df.get("High", df["Close"])
df["Low"] = df.get("Low", df["Close"])
df["Return"] = df["Close"].pct_change().fillna(0)

high, low, close = df["High"], df["Low"], df["Close"]
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi().bfill()
df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

# ----------------------------------------------------------
# 🔹 Prognose berechnen
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
# 🔹 Ergebnis speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgaspreis: {round(last_close,3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
)

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("✅ Ergebnis in result.txt gespeichert.")

# ----------------------------------------------------------
# 🔹 Änderungserkennung (>10 % oder Trendwechsel)
# ----------------------------------------------------------
def get_previous_info(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        m_prob = re.search(r'Wahrscheinlichkeit steigend:\s*([0-9.]+)', text)
        m_trend = re.search(r'Trend:\s*(Steigend|Fallend)', text)
        prob = float(m_prob.group(1)) if m_prob else None
        tr = m_trend.group(1) if m_trend else None
        return prob, tr
    return None, None

prev_prob, prev_trend = get_previous_info(PREVIOUS_FILE)
change_triggered = False

if prev_prob is not None:
    diff = abs(trend_prob - prev_prob) / prev_prob * 100 if prev_prob != 0 else 0
    print(f"🔸 Änderung: {diff:.2f}% (Trend vorher: {prev_trend} → jetzt: {trend})")

    if diff > 10 or prev_trend != ("Steigend" if trend_prob >= 50 else "Fallend"):
        print("⚠️ Signifikante Änderung oder Trendwechsel erkannt!")
        change_triggered = True
else:
    print("ℹ️ Kein Vergleichswert vorhanden (erster Lauf).")

with open(PREVIOUS_FILE, "w", encoding="utf-8") as f:
    f.write(msg)
print("💾 previous_result.txt aktualisiert.")
