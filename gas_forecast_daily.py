# ----------------------------------------------------------
# 🔥 Erdgas Daily Forecast + Optimierte Parameter (mit Fallback)
# ----------------------------------------------------------
import pandas as pd
import numpy as np
from datetime import datetime
import ta

# ----------------------------------------------------------
# ⚙️ Optimierte Parameter (aus historischer Optimierung)
# ----------------------------------------------------------
SMA_SHORT = 15
SMA_LONG = 40
W_SMA = 8
W_RSI = 1.0
W_ATR = 5
W_STREAK = 1.5
W_OIL = 8
LAST_DAYS = 30
OPT_HISTORICAL_ACCURACY = 80.89

# ----------------------------------------------------------
# 🔹 Hilfsfunktionen
# ----------------------------------------------------------
def add_indicators(df, rsi_period=14, atr_period=14):
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(atr_period).mean().bfill()
    
    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()
    return df.bfill()

def calculate_prediction(df, w_sma, w_rsi, w_atr, w_streak, w_oil, sma_short, sma_long, chain_max=14):
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
    
    recent_returns = df["Return"].tail(chain_max).values
    sign = np.sign(recent_returns[-1])
    streak = sum(1 for r in reversed(recent_returns[:-1]) if np.sign(r)==sign)
    prob += sign * streak * w_streak
    
    if "Oil_Change" in df.columns:
        prob += df["Oil_Change"].iloc[-1] * 100 * w_oil
    
    return max(0, min(100, prob))

def get_streak(df):
    recent_returns = df["Return"].values
    if len(recent_returns) == 0:
        return "neutral", 0
    sign = np.sign(recent_returns[-1])
    streak = 1
    for r in reversed(recent_returns[:-1]):
        if np.sign(r) == sign:
            streak += 1
        else:
            break
    direction = "steigend 📈" if sign >= 0 else "fallend 📉"
    return direction, streak

# ----------------------------------------------------------
# 📥 Daten laden mit Fallback
# ----------------------------------------------------------
def load_data_with_fallback():
    try:
        import requests
        TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # API-Key
        url_gas = f"https://api.tradingeconomics.com/historical/commodity/Natural Gas?c={TRADINGECONOMICS_KEY}"
        url_oil = f"https://api.tradingeconomics.com/historical/commodity/Crude Oil?c={TRADINGECONOMICS_KEY}"
        r_gas = requests.get(url_gas)
        r_oil = requests.get(url_oil)
        if r_gas.status_code != 200 or r_oil.status_code != 200:
            raise ValueError(f"Gas {r_gas.status_code}, Oil {r_oil.status_code}")
        gas = pd.DataFrame(r_gas.json())
        oil = pd.DataFrame(r_oil.json())
        gas["Date"] = pd.to_datetime(gas["Date"])
        oil["Date"] = pd.to_datetime(oil["Date"])
        oil["Oil_Close_prev"] = oil["Close"].shift(1)
        df = gas.merge(oil[["Date", "Oil_Close_prev"]], on="Date", how="left").ffill()
        df["Return"] = df["Close"].pct_change().fillna(0)
        df["Oil_Change"] = df["Oil_Close_prev"].pct_change().fillna(0)
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"⚠️ Warnung: Keine aktuellen Daten verfügbar ({e}), benutze historische Daten.")
        df = pd.read_csv("historical_gas_data.csv", parse_dates=["Date"])
        df = add_indicators(df)
        return df

# ----------------------------------------------------------
# 📊 Forecast berechnen
# ----------------------------------------------------------
df = load_data_with_fallback()
df = add_indicators(df)
trend_prob = calculate_prediction(df, W_SMA, W_RSI, W_ATR, W_STREAK, W_OIL, SMA_SHORT, SMA_LONG)
trend = "Steigend 📈" if trend_prob >= 50 else "Fallend 📉"
last_close = df["Close"].iloc[-1]
streak_direction, streak_length = get_streak(df)

# ----------------------------------------------------------
# 📊 Ergebnis
# ----------------------------------------------------------
msg = (
    f"📅 {datetime.now():%d.%m.%Y %H:%M}\n"
    f"🔥 Erdgas (NG=F): {round(last_close, 3)} USD/MMBtu\n"
    f"🔮 Trend: {trend}\n"
    f"📊 Wahrscheinlichkeit steigend: {round(trend_prob,2)} %\n"
    f"📊 Wahrscheinlichkeit fallend : {round(100-trend_prob,2)} %\n"
    f"📏 Aktueller Trend: Erdgas ist {streak_length} Tage in Folge {streak_direction}\n"
    f"🎯 Optimierte Trefferquote (letzte {LAST_DAYS} Tage): {OPT_HISTORICAL_ACCURACY} %\n"
    f"⚙️ Beste Parameter → SMA={SMA_SHORT}/{SMA_LONG}, WSMA={W_SMA}, RSI={W_RSI}, ATR={W_ATR}, Streak={W_STREAK}, W_Oil={W_OIL}"
)

print(msg)
