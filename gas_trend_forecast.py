  # ----------------------------------------------------------
# 🔥 Erdgas Trend Forecast + Rolling Accuracy (fehlerfrei)
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# ⚙️ Parameter
# ----------------------------------------------------------
SYMBOL = "NG=F"
OIL_SYMBOL = "CL=F"

ATR_PERIOD = 14
RSI_PERIOD = 14
SMA_SHORT = 15
SMA_LONG = 40
CHAIN_MAX = 14
ROLL_WINDOW = 30  # Rolling Accuracy Fenster

W_SMA = 5
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.0
OIL_WEIGHT = 5

END = datetime.now()
START = END - timedelta(days=3*365)

# ----------------------------------------------------------
# 📥 Daten laden
# ----------------------------------------------------------
def load_data(ticker, oil_ticker):
    gas = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    oil = yf.download(oil_ticker, start=START, end=END, auto_adjust=True, progress=False)

    gas = gas.rename(columns=str.title).reset_index()
    gas["Return"] = gas["Close"].pct_change().fillna(0)
    gas["Date"] = pd.to_datetime(gas["Date"])

    oil = oil.rename(columns=str.title).reset_index()
    oil["Oil_Close_prev"] = oil["Close"].shift(1)

    gas = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
    gas["Oil_Change"] = gas["Oil_Close_prev"].pct_change().fillna(0)
    return gas

df = load_data(SYMBOL, OIL_SYMBOL)
print(f"✅ Loaded {len(df)} days of Natural Gas data.")

# ----------------------------------------------------------
# 📊 Indikatoren
# ----------------------------------------------------------
def add_indicators(df):
    df = df.copy()
    for col in ["High", "Low", "Close"]:
        series = df.get(col, None)
        if series is None or (isinstance(series, pd.Series) and (series.empty or series.isnull().all())):
            df[col] = df["Close"]
        elif isinstance(series, pd.DataFrame):
            df[col] = series.iloc[:,0]

    # ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean().bfill()

    # RSI 1D fix
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:,0]
    df["RSI"] = ta.momentum.RSIIndicator(close_series, window=RSI_PERIOD).rsi()

    return df.bfill()

df = add_indicators(df)

# ----------------------------------------------------------
# 🔮 Prognoseberechnung
# ----------------------------------------------------------
def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long):
    if len(df) < sma_long:
        return 50
    df = df.copy()
    df["sma_short"] = df["Close"].rolling(sma_short).mean()
    df["sma_long"] = df["Close"].rolling(sma_long).mean()

    prob = 50

    # SMA Trend
    sma_short_last = df["sma_short"].iloc[-1]
    sma_long_last  = df["sma_long"].iloc[-1]
    prob += w_sma if sma_short_last > sma_long_last else -w_sma

    # RSI
    rsi_last = df["RSI"].iloc[-1]
    prob += (rsi_last - 50) * w_rsi / 10

    # ATR-weighted move
    atr_last = df["ATR"].iloc[-1]
    if atr_last > 0:
        daily_move = df["Return"].iloc[-1]
        prob += np.tanh((daily_move / atr_last) * 2) * w_atr

    # Trendserie / Streak
    recent_returns = df["Return"].tail(CHAIN_MAX).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r)==sign)
    prob += sign * streak * w_streak

    # Ölpreis Einfluss
    oil_change_last = df["Oil_Change"].iloc[-1]
    prob += oil_change_last * 100 * w_oil

    return max(0, min(100, prob))

# ----------------------------------------------------------
# 🔹 Aktueller Trend
# ----------------------------------------------------------
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, OIL_WEIGHT, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]

# ----------------------------------------------------------
# 🔹 Trendserie
# ----------------------------------------------------------
def get_streak(df):
    recent_returns = df["Return"].tail(30).values
    up = recent_returns[-1] > 0
    streak = 1
    for r in reversed(recent_returns[:-1]):
        if (r > 0 and up) or (r < 0 and not up):
            streak += 1
        else:
            break
    direction = "gestiegen 📈" if up else "gefallen 📉"
    return direction, streak

streak_direction, streak_length = get_streak(df)

# ----------------------------------------------------------
# 📊 Rolling Accuracy
# ----------------------------------------------------------
def rolling_accuracy(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long, window=ROLL_WINDOW):
    acc_list = []
    dates = []

    for i in range(window, len(df)):
        correct = 0
        for j in range(i-window, i):
            df_slice = df.iloc[:j+1].copy()
            prob = calculate_prediction(df_slice, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long)
            predicted_up = prob >= 50
            actual_up = df["Return"].iloc[j] > 0
            if predicted_up == actual_up:
                correct += 1
        acc_list.append(correct / window * 100)
        dates.append(df["Date"].iloc[i])

    return pd.DataFrame({"Date": dates, "Accuracy": acc_list})

rolling_acc_df = rolling_accuracy(df, W_SMA, W_RSI, W_ATR, W_STREAK, OIL_WEIGHT, SMA_SHORT, SMA_LONG)
print("✅ Rolling Accuracy berechnet.")

# ----------------------------------------------------------
# 📊 Ergebnis ausgeben & speichern
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"📈 Erdgas: {round(last_close,2)} $\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"📏 Aktueller Trend: Erdgas ist {streak_length} Tage in Folge {streak_direction}\n"
)
print(msg)

with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("📁 Ergebnis in result.txt gespeichert ✅")

# ----------------------------------------------------------
# 📈 Rolling Accuracy Plot
# ----------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(rolling_acc_df["Date"], rolling_acc_df["Accuracy"], label="Rollierende Trefferquote", color="blue")
plt.axhline(50, color="red", linestyle="--", label="Zufall (50%)")
plt.title("Rolling Accuracy der Erdgas-Vorhersage")
plt.xlabel("Datum")
plt.ylabel("Trefferquote (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rolling_accuracy.png")
plt.show()
print("📁 Rolling Accuracy Diagramm gespeichert als rolling_accuracy.png ✅")
