# ----------------------------------------------------------
# 🔥 Erdgas Tagesforecast + automatische Ausgabe in result.txt
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
from itertools import product
import time

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # API-Key einfügen
SYMBOL_GAS = "Natural Gas"
SYMBOL_OIL = "Crude Oil"

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14
ROLL_WINDOW = 30

# Optimierte Parameter (Beispiel)
SMA_SHORT = 15
SMA_LONG  = 40
W_SMA     = 8
W_RSI     = 1.0
W_ATR     = 5
W_STREAK  = 1.5
W_OIL     = 8

# Historie: letzte 4 Jahre
END = datetime.now()
START = END - timedelta(days=4*365)

# ----------------------------------------------------------
# 📥 Daten von TradingEconomics laden (mit 3 Versuchen)
# ----------------------------------------------------------
def load_data_te(max_retries=3, delay=5):
    for attempt in range(1, max_retries+1):
        try:
            print(f"⏳ Lade Daten von TradingEconomics (Versuch {attempt}) ...")
            url_gas = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
            url_oil = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"

            r_gas = requests.get(url_gas)
            r_oil = requests.get(url_oil)

            if r_gas.status_code != 200 or r_oil.status_code != 200:
                print(f"⚠️ Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")
                if attempt < max_retries:
                    print(f"⏳ Warte {delay} Sekunden und versuche erneut...")
                    time.sleep(delay)
                    continue
                else:
                    raise ValueError(f"Fehler nach {max_retries} Versuchen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

            gas = pd.DataFrame(r_gas.json())
            oil = pd.DataFrame(r_oil.json())

            gas = gas.rename(columns=str.title)
            oil = oil.rename(columns=str.title)

            gas["Date"] = pd.to_datetime(gas["Date"])
            oil["Date"] = pd.to_datetime(oil["Date"])
            oil["Oil_Close_prev"] = oil["Close"].shift(1)

            df = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
            df["Return"] = df["Close"].pct_change().fillna(0)
            df["Oil_Change"] = df["Oil_Close_prev"].pct_change().fillna(0)

            # Nur die letzten 4 Jahre
            df = df[df["Date"] >= START].reset_index(drop=True)
            return df
        except Exception as e:
            last_exception = e
    raise last_exception

# ----------------------------------------------------------
# 📊 Indikatoren berechnen
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

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"].values, window=RSI_PERIOD).rsi()
    return df.bfill()

# ----------------------------------------------------------
# 🔮 Trendvorhersage
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    prob = 50
    prob += w_sma if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -w_sma
    prob += (df["RSI"].iloc[-1]-50) * w_rsi / 10

    atr_last = df["ATR"].iloc[-1]
    if atr_last>0:
        prob += np.tanh((df["Return"].iloc[-1]/atr_last)*2)*w_atr

    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r)==sign)
    prob += sign*streak*w_streak

    prob += df["Oil_Change"].iloc[-1]*100*w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 📏 Aktueller Trend
# ----------------------------------------------------------
def get_streak(df):
    returns = df["Return"].values
    last_sign = np.sign(returns[-1])
    streak = 1
    for r in reversed(returns[:-1]):
        if np.sign(r) == last_sign:
            streak += 1
        else:
            break
    direction = "steigend 📈" if last_sign>0 else "fallend 📉"
    return direction, streak

# ----------------------------------------------------------
# 📊 Rolling Accuracy
# ----------------------------------------------------------
def rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long, w_oil, window=ROLL_WINDOW):
    acc_list = []
    for i in range(window, len(df)):
        correct = 0
        for j in range(i-window, i):
            df_slice = df.iloc[:j+1].copy()
            prob = calculate_prediction(df_slice, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long)
            predicted_up = prob >= 50
            actual_up = df["Return"].iloc[j] > 0
            if predicted_up == actual_up:
                correct += 1
        acc_list.append(correct/window*100)
    return acc_list

# ----------------------------------------------------------
# 📥 Hauptprogramm
# ----------------------------------------------------------
try:
    df = load_data_te()
except Exception as e:
    msg = f"⚠️ Warnung: Keine aktuellen Daten verfügbar, Fehler: {e}"
    print(msg)
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    exit(1)

df = add_indicators(df)

# Trendvorhersage
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]
streak_direction, streak_length = get_streak(df)
rolling_acc = rolling_accuracy(df, W_SMA, W_RSI, W_ATR, W_STREAK, SMA_SHORT, SMA_LONG, W_OIL)

# Ergebnis zusammenstellen
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas ({SYMBOL_GAS}): {round(last_close,3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"📏 Aktueller Trend: Erdgas ist {streak_length} Tage in Folge {streak_direction}\n"
    f"🎯 Rolling Accuracy (Median): {round(np.median(rolling_acc),2)} %, Min: {round(np.min(rolling_acc),2)} %, Max: {round(np.max(rolling_acc),2)} %, Std: {round(np.std(rolling_acc),2)} %\n"
    f"⚙️ Optimierte Parameter → SMA={SMA_SHORT}/{SMA_LONG}, W_SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, Oil_Weight={W_OIL}"
)

# Ergebnis ausgeben & speichern
print(msg)
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert ✅")
