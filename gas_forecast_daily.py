#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
import ta

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

# ----------------------------------------------------------
# 🔹 Historische Daten laden oder neu anlegen
# ----------------------------------------------------------
try:
    df = pd.read_csv(HIST_FILE, parse_dates=["Date"])
    print(f"✅ Historische Daten geladen: {len(df)} Tage")
except FileNotFoundError:
    print("⚠️ Keine historische Datei gefunden. Neue wird erstellt.")
    df = pd.DataFrame(columns=["Date", "Close", "High", "Low"])

# ----------------------------------------------------------
# 🔹 Preisabruffunktionen
# ----------------------------------------------------------
def get_tradingeconomics_price():
    try:
        url = f"https://api.tradingeconomics.com/markets/commodities?c=DEIN_KEY_HIER"
        data = requests.get(url, timeout=10).json()
        for item in data:
            if "Natural Gas" in item.get("name", "") or item.get("symbol") == "NATGAS":
                price = float(item.get("last", 0))
                if price > 0:
                    return price
    except:
        return None
    return None

def get_eia_price():
    try:
        url = f"https://api.eia.gov/v2/natural-gas/pri/whd/data/?api_key=DEIN_EIA_KEY_HIER&frequency=daily&data[0]=value&facets[series][]=NG.RNGWHHD.D&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=1"
        data = requests.get(url, timeout=10).json()
        price = float(data["response"]["data"][0]["value"])
        return price
    except:
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
            return min(candidates)
    except:
        return None
    return None

# ----------------------------------------------------------
# 🔹 Primärquelle Yahoo Finance NG=F mit Fallback
# ----------------------------------------------------------
PRIMARY_SOURCE = "Yahoo Finance"
FALLBACK_USED = False
today_price = None

try:
    import yfinance as yf
    gas = yf.Ticker("NG=F")
    price = gas.info.get("regularMarketPrice")
    if price and price > 0:
        today_price = float(price)
        source_used = PRIMARY_SOURCE
        print(f"✅ Preis von {PRIMARY_SOURCE}: {today_price} USD/MMBtu")
except Exception as e:
    print(f"⚠️ {PRIMARY_SOURCE} nicht verfügbar: {e}")
    FALLBACK_USED = True

if today_price is None:
    for src in [get_tradingeconomics_price, get_eia_price, get_finanzen_price]:
        today_price = src()
        if today_price:
            source_used = "Fallback"
            FALLBACK_USED = True
            break

if not today_price:
    raise SystemExit("❌ Kein gültiger Preis gefunden (alle Quellen fehlgeschlagen).")

# ----------------------------------------------------------
# 🔹 Historie erweitern
# ----------------------------------------------------------
today = pd.Timestamp(datetime.now().date())
if not ((df["Date"] == today).any()):
    df.loc[len(df)] = [today, today_price, today_price, today_price]
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
# 🔹 Vorheriges Ergebnis einlesen
# ----------------------------------------------------------
def get_previous_info(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        m_prob = re.search(r"Wahrscheinlichkeit steigend:\s*([0-9.]+)", text)
        m_trend = re.search(r'Trend:\s*(Steigend|Fallend)', text)
        prob = float(m_prob.group(1)) if m_prob else None
        tr = m_trend.group(1) if m_trend else None
        return prob, tr

prev_prob, prev_trend = get_previous_info(PREVIOUS_FILE)
diff_percent = 0.0
if prev_prob is not None:
    diff_percent = ((trend_prob - prev_prob) / abs(prev_prob)) * 100 if prev_prob != 0 else 0

source_warning = " (Achtung nicht gleiche Quelle)" if FALLBACK_USED else ""

# ----------------------------------------------------------
# 🔹 Ergebnisstring erstellen
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgaspreis: {round(last_close,3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob,2)} %\n"
)

if diff_percent is not None:
    sign = "+" if diff_percent >= 0 else "−"
    msg += f"📈 Unterschied zur letzten Berechnung: {sign}{abs(round(diff_percent,2))} %{source_warning}\n"

# ----------------------------------------------------------
# 🔹 Textdateien speichern (korrigierte Version)
# ----------------------------------------------------------

# 1. result.txt wird IMMER überschrieben
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)

# 2. previous_result.txt wird NUR überschrieben, wenn sich die Wahrscheinlichkeit geändert hat
update_previous = False

if prev_prob is None:
    update_previous = True
else:
    # Nur speichern, wenn Unterschied vorhanden ist
    if round(prev_prob, 2) != round(trend_prob, 2):
        update_previous = True

if update_previous:
    with open(PREVIOUS_FILE, "w", encoding="utf-8") as f:
        f.write(msg)

# ----------------------------------------------------------
# 🔹 CSV-Log im Repo-Root
# ----------------------------------------------------------
LOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gas_forecast_log.csv")

row = {
    "Datum": datetime.now().strftime("%d.%m.%Y"),
    "Uhrzeit": datetime.now().strftime("%H:%M"),
    "Erdgaspreis_USD/MMBtu": round(last_close, 3),
    "Trend": "Steigend" if "Steigend" in trend else "Fallend",
    "Wahrscheinlichkeit_steigend_%": round(trend_prob, 2),
    "Wahrscheinlichkeit_fallend_%": round(100 - trend_prob, 2),
    "Unterschied_%": round(diff_percent, 2)
}

if not os.path.exists(LOG_CSV):
    pd.DataFrame([row]).to_csv(LOG_CSV, index=False, encoding="utf-8")
else:
    df_log = pd.read_csv(LOG_CSV)
    df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    df_log.to_csv(LOG_CSV, index=False, encoding="utf-8")

