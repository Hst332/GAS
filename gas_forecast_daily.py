# ----------------------------------------------------------
# 🔥 Erdgas Daily Forecast + Trendwahrscheinlichkeit
# ----------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # Hier deinen Key einfügen

SYMBOL_GAS = "Natural Gas"
SYMBOL_OIL = "Crude Oil"

ATR_PERIOD = 14
RSI_PERIOD = 14
CHAIN_MAX = 14

# Optimierte Parameter (z.B. aus vorheriger Optimierung)
SMA_SHORT = 15
SMA_LONG  = 40
W_SMA     = 8
W_RSI     = 1.0
W_ATR     = 5
W_STREAK  = 1.5
W_OIL     = 8

# ----------------------------------------------------------
# 📥 Daten laden von TradingEconomics
# ----------------------------------------------------------
def load_data_te():
    print("⏳ Lade aktuelle Daten von TradingEconomics ...")
    
    url_gas = f"https://api.tradingeconomics.com/historical/commodity/Natural%20Gas?c={TRADINGECONOMICS_KEY}"
    url_oil = f"https://api.tradingeconomics.com/historical/commodity/Crude%20Oil?c={TRADINGECONOMICS_KEY}"

    r_gas = requests.get(url_gas)
    r_oil = requests.get(url_oil)

    if r_gas.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen der Gasdaten: {r_gas.status_code}")

    gas = pd.DataFrame(r_gas.json())
    gas = gas.rename(columns=str.title)
    gas["Date"] = pd.to_datetime(gas["Date"])
    gas["Return"] = gas["Close"].pct_change().fillna(0)

    # Öl optional
    if r_oil.status_code != 200:
        print("⚠️ Kein Ölpreis verfügbar, fahre nur mit Gasdaten fort ...")
        gas["Oil_Close_prev"] = 0
        gas["Oil_Change"] = 0
        return gas

    oil = pd.DataFrame(r_oil.json())
    oil = oil.rename(columns=str.title)
    oil["Date"] = pd.to_datetime(oil["Date"])
    oil["Oil_Close_prev"] = oil["Close"].shift(1)
    gas = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    gas["Oil_Change"] = gas["Oil_Close_prev"].pct_change().fillna(0)

    return gas

# ----------------------------------------------------------
# 📊 Indikatoren berechnen
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()

    # ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=RSI_PERIOD).rsi()
    return df.bfill()

# ----------------------------------------------------------
# 🔮 Trendvorhersage für heute
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50

    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    prob = 50
    prob += w_sma if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1] else -w_sma
    prob += (df["RSI"].iloc[-1] - 50) * w_rsi / 10

    atr_last = df["ATR"].iloc[-1]
    if atr_last > 0:
        prob += np.tanh((df["Return"].iloc[-1]/atr_last)*2) * w_atr

    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r) == sign)
    prob += sign * streak * w_streak

    prob += df["Oil_Change"].iloc[-1] * 100 * w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 📏 Aktuelle Trend-Streak
# ----------------------------------------------------------
def get_streak(df):
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    length = 1
    for r in reversed(recent_returns[:-1]):
        if np.sign(r) == sign:
            length += 1
        else:
            break
    direction = "steigend 📈" if sign > 0 else "fallend 📉"
    return direction, length

# ----------------------------------------------------------
# 📊 Rolling Accuracy (letzte Tage)
# ----------------------------------------------------------
def rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, sma_short, sma_long, w_oil=0, window=30):
    acc_list = []
    for i in range(window, len(df)):
        correct = 0
        for j in range(i - window, i):
            df_slice = df.iloc[:j+1]
            prob = calculate_prediction(df_slice, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long)
            predicted_up = prob >= 50
            actual_up = df["Return"].iloc[j] > 0
            if predicted_up == actual_up:
                correct += 1
        acc_list.append(correct / window * 100)
    return acc_list

# ----------------------------------------------------------
# 🔹 Main
# ----------------------------------------------------------
df = load_data_te()
df = add_indicators(df)

trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]
streak_direction, streak_length = get_streak(df)
rolling_acc = rolling_accuracy(df, W_SMA, W_RSI, W_ATR, W_STREAK, SMA_SHORT, SMA_LONG, W_OIL)

# ----------------------------------------------------------
# 📊 Ergebnis ausgeben
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas: {round(last_close, 3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob, 2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100 - trend_prob, 2)} %\n"
    f"📏 Aktueller Trend: Erdgas ist {streak_length} Tage in Folge {streak_direction}\n"
    f"🎯 Optimierte Trefferquote (Median der letzten {len(rolling_acc)} Tage): {round(np.median(rolling_acc), 2)} %\n"
    f"⚙️ Beste Parameter → SMA={SMA_SHORT}/{SMA_LONG}, W_SMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, W_Oil={W_OIL}"
)

print(msg)
