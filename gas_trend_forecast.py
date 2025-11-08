# ----------------------------------------------------------
# 🔥 Erdgas Trend Forecast + Optimierte Parameter mit TradingEconomics (4 Jahre)
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt
from itertools import product

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

# Suchbereich für Optimierung (leicht reduziert für Geschwindigkeit)
SMA_SHORT_RANGE = [10, 15, 20]
SMA_LONG_RANGE  = [30, 40, 50]
W_SMA_RANGE     = [5, 8]
W_RSI_RANGE     = [0.8, 1.0]
W_ATR_RANGE     = [4, 5]
W_STREAK_RANGE  = [1.5, 2.0]
OIL_WEIGHT_RANGE= [5, 8]

# Historie: letzte 4 Jahre
END = datetime.now()
START = END - timedelta(days=4*365)

# ----------------------------------------------------------
# 📥 Daten laden von TradingEconomics
# ----------------------------------------------------------
def load_data_te():
    print("⏳ Lade Daten von TradingEconomics ...")
    url_gas = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_GAS}?c={TRADINGECONOMICS_KEY}"
    url_oil = f"https://api.tradingeconomics.com/historical/commodity/{SYMBOL_OIL}?c={TRADINGECONOMICS_KEY}"

    r_gas = requests.get(url_gas)
    r_oil = requests.get(url_oil)

    if r_gas.status_code != 200 or r_oil.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

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

df = load_data_te()
print(f"✅ Loaded {len(df)} days of data ({df['Date'].iloc[0]} → {df['Date'].iloc[-1]})")

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()

    # ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

    # RSI
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:,0]
    df["RSI"] = ta.momentum.RSIIndicator(close_series, window=RSI_PERIOD).rsi()

    return df.bfill()

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Vorhersageberechnung
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
# 📊 Rolling Accuracy
# ----------------------------------------------------------
def rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long, window=ROLL_WINDOW, step=2):
    acc_list = []
    for i in range(window, len(df), step):
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
# 🔍 Parameteroptimierung
# ----------------------------------------------------------
best_mean = -1
best_params = None

for sma_short, sma_long, w_sma, w_rsi, w_atr, w_streak, w_oil in product(
        SMA_SHORT_RANGE, SMA_LONG_RANGE, W_SMA_RANGE, W_RSI_RANGE, W_ATR_RANGE, W_STREAK_RANGE, OIL_WEIGHT_RANGE):
    if sma_short >= sma_long:
        continue
    acc = rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long, step=5)
    mean_acc = np.mean(acc)
    if mean_acc > best_mean:
        best_mean = mean_acc
        best_params = (sma_short, sma_long, w_sma, w_rsi, w_atr, w_streak, w_oil)

print("✅ Optimierung abgeschlossen")
print(f"Beste Parameter: SMA={best_params[0]}/{best_params[1]}, W_SMA={best_params[2]}, W_RSI={best_params[3]}, W_ATR={best_params[4]}, Streak={best_params[5]}, Oil_Weight={best_params[6]}")
print(f"Durchschnittliche Rolling Accuracy: {best_mean:.2f} %")

# ----------------------------------------------------------
# 📊 Rolling Accuracy mit besten Parametern
# ----------------------------------------------------------
best_acc = rolling_accuracy(df, *best_params, step=1)
print(f"Median: {np.median(best_acc):.2f} %")
print(f"Minimum: {np.min(best_acc):.2f} %")
print(f"Maximum: {np.max(best_acc):.2f} %")
print(f"Std-Abw.: {np.std(best_acc):.2f} %")

# ----------------------------------------------------------
# 📈 Plot
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(df["Date"].tail(len(best_acc)), best_acc, label="Rollierende Trefferquote", color="blue")
plt.axhline(50, color="red", linestyle="--", label="Zufall (50%)")
plt.title("Rolling Accuracy der Erdgas-Vorhersage (Optimiert, 4 Jahre)")
plt.xlabel("Datum")
plt.ylabel("Trefferquote (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rolling_accuracy_4years.png")
plt.show()
print("📁 Plot gespeichert als rolling_accuracy_4years.png ✅")
