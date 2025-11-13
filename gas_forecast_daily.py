# ----------------------------------------------------------
# 🌐 GAS FORECAST DAILY (Henry Hub Spotpreis)
# Multi-Source Version: TradingEconomics → EIA → Yahoo → finanzen.net
# ----------------------------------------------------------

import os
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
import ta
import yfinance as yf

# ----------------------------------------------------------
# ⚙️ Einstellungen
# ----------------------------------------------------------
HIST_FILE = "gas_history.csv"
PREVIOUS_FILE = "previous_result.txt"
TRADINGECONOMICS_KEY = os.getenv("TRADINGECONOMICS_KEY", "DEIN_KEY_HIER")
EIA_API_KEY = os.getenv("EIA_API_KEY", "DEIN_EIA_KEY_HIER")

SMA_SHORT = 15
SMA_LONG = 40
W_SMA = 8
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.5
ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14

# ----------------------------------------------------------
# 📜 Historische Henry-Hub-Daten abrufen (60 Tage)
# ----------------------------------------------------------
def get_initial_history(days=60):
    try:
        gas = yf.Ticker("NG=F")
        hist = gas.history(period=f"{days}d")
        if hist.empty:
            raise ValueError("Keine historischen Daten von Yahoo Finance erhalten.")
        df_hist = hist[["Close", "High", "Low"]].reset_index()
        df_hist.rename(columns={"Date": "Date"}, inplace=True)
        df_hist["Date"] = pd.to_datetime(df_hist["Date"]).dt.date
        print(f"✅ {len(df_hist)} historische Gasdaten geladen (Yahoo Finance).")
        return df_hist
    except Exception as e:
        print(f"⚠️ Historische Daten konnten nicht geladen werden: {e}")
        return pd.DataFrame(columns=["Date", "Close", "High", "Low"])

# ----------------------------------------------------------
# 🔹 Historische Daten laden oder neu anlegen
# ----------------------------------------------------------
try:
    df = pd.read_csv(HIST_FILE, parse_dates=["Date"])
    print(f"✅ Historische Daten geladen: {len(df)} Tage")
except FileNotFoundError:
    print("⚠️ Keine historische Datei gefunden. Lade initiale Daten …")
    df = get_initial_history(60)
    df.to_csv(HIST_FILE, index=False)

# ----------------------------------------------------------
# 🔹 Multi-Source Preisabruf
# ----------------------------------------------------------
def get_tradingeconomics_price():
    try:
        url = f"https://api.tradingeconomics.com/markets/commodities?c={TRADINGECONOMICS_KEY}"
        data = requests.get(url, timeout=10).json()
        for item in data:
            if "Natural Gas" in item.get("name", "") or item.get("symbol") == "NATGAS":
                price = float(item.get("last", 0))
                if price > 0:
                    print(f"✅ Preis von TradingEconomics: {price} USD/MMBtu")
                    return price
    except Exception as e:
        print(f"⚠️ TradingEconomics nicht verfügbar: {e}")
    return None

def get_eia_price():
    try:
        url = f"https://api.eia.gov/v2/natural-gas/pri/whd/data/?api_key={EIA_API_KEY}&frequency=daily&data[0]=value&facets[series][]=NG.RNGWHHD.D&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=1"
        data = requests.get(url, timeout=10).json()
        price = float(data["response"]["data"][0]["value"])
        if price > 0:
            print(f"✅ Preis von EIA: {price} USD/MMBtu")
            return price
    except Exception as e:
        print(f"⚠️ EIA API nicht verfügbar: {e}")
    return None

def get_yahoo_price():
    try:
        gas = yf.Ticker("NG=F")
        price = gas.info.get("regularMarketPrice")
        if price and price > 0:
            print(f"✅ Preis von Yahoo Finance (Future): {price} USD/MMBtu")
            return price
    except Exception as e:
        print(f"⚠️ Yahoo Finance nicht verfügbar: {e}")
    return None

def get_finanzen_price():
    try:
        url = "https://www.finanzen.net/rohstoffe/erdgas-preis-natural-gas"
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        candidates = []
        for tag in soup.find_all(text=re.compile(r"([0-9]+,[0-9]+)\s*USD")):
            if any(x in tag for x in ["Erdgas", "Natural Gas", "MMBtu"]):
                val = re.search(r"([0-9]+,[0-9]+)", tag)
                if val:
                    price = float(val.group(1).replace(",", "."))
                    if 1 < price < 50:
                        candidates.append(price)
        if candidates:
            price = min(candidates)
            print(f"✅ Preis von finanzen.net: {price} USD/MMBtu")
            return price
    except Exception as e:
        print(f"⚠️ finanzen.net nicht verfügbar: {e}")
    return None

sources = [get_tradingeconomics_price, get_eia_price, get_yahoo_price, get_finanzen_price]
today_price = None
for src in sources:
    today_price = src()
    if today_price:
        break
if not today_price:
    raise SystemExit("❌ Kein gültiger Preis gefunden (alle Quellen fehlgeschlagen).")

# ----------------------------------------------------------
# 🔹 Preis speichern / Historie erweitern
# ----------------------------------------------------------
today = pd.Timestamp(datetime.now().date())
if not ((df["Date"] == today).any()):
    df.loc[len(df)] = [today, today_price, today_price, today_price]
    print(f"💾 Neuer Datensatz: {today_price} USD/MMBtu ({today.date()})")
df.to_csv(HIST_FILE, index=False)

# ----------------------------------------------------------
# 🔹 Indikatoren berechnen
# ----------------------------------------------------------
df["Return"] = df["Close"].pct_change().fillna(0)
high, low, close = df["High"], df["Low"], df["Close"]
tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi().bfill()
df["sma_short"] = df["Close"].rolling(SMA_SHORT).mean()
df["sma_long"] = df["Close"].rolling(SMA_LONG).mean()

# ----------------------------------------------------------
# 🔹 Prognosefunktion
# ----------------------------------------------------------
def calculate_prediction(df):
    # Sicherheitsprüfungen
    if df["Close"].count() < max(SMA_LONG, ATR_PERIOD, RSI_PERIOD):
        print("⚠️ Zu wenige Daten für Berechnung – neutraler Wert verwendet.")
        return 50.0
    if df[["sma_short", "sma_long", "RSI", "ATR"]].iloc[-1].isnull().any():
        print("⚠️ Unvollständige Indikatoren – neutraler Wert verwendet.")
        return 50.0

    prob = 50
    prob += W_SMA if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -W_SMA
    prob += (df["RSI"].iloc[-1] - 50) * W_RSI / 10
    prob += np.tanh(df["Return"].iloc[-1] / df["ATR"].iloc[-1]) * W_ATR
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1]) if len(recent_returns) > 0 else 0
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * W_STREAK
    return prob  # keine Begrenzung

trend_prob = calculate_prediction(df)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 🔹 Unterschied zur vorherigen Berechnung (%)
# ----------------------------------------------------------
diff_percent = None
if os.path.exists(PREVIOUS_FILE):
    with open(PREVIOUS_FILE, "r", encoding="utf-8") as f:
        prev_text = f.read()
        m_prev = re.search(r"Wahrscheinlichkeit steigend:\s*([0-9.\-]+)", prev_text)
        if m_prev:
            prev_prob = float(m_prev.group(1))
            if prev_prob != 0:
                diff_percent = ((trend_prob - prev_prob) / abs(prev_prob)) * 100

# ----------------------------------------------------------
# 🔹 Ergebnis speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgaspreis: {round(last_close,3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob,2)} %\n"
)
if diff_percent is not None:
    msg += f"📈 Unterschied zur letzten Berechnung: {round(diff_percent,2)} %\n"

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("✅ Ergebnis in result.txt gespeichert.\n")
print(msg)

# ----------------------------------------------------------
# 🔹 Trendänderungserkennung
# ----------------------------------------------------------
def get_previous_info(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        m_prob = re.search(r"Wahrscheinlichkeit steigend:\s*([0-9.]+)", text)
        m_trend = re.search(r"Trend:\s*(Steigend|Fallend)", text)
        prob = float(m_prob.group(1)) if m_prob else None
        tr = m_trend.group(1) if m_trend else None
        return prob, tr

prev_prob, prev_trend = get_previous_info(PREVIOUS_FILE)
if prev_prob is not None:
    diff = abs(trend_prob - prev_prob) / prev_prob * 100 if prev_prob != 0 else 0
    if diff > 10 or prev_trend != ("Steigend" if trend_prob >= 50 else "Fallend"):
        print("⚠️ Signifikante Änderung oder Trendwechsel erkannt!")
else:
    print("ℹ️ Kein Vergleichswert vorhanden (erster Lauf).")

with open(PREVIOUS_FILE, "w", encoding="utf-8") as f:
    f.write(msg)
print("💾 previous_result.txt aktualisiert.")
