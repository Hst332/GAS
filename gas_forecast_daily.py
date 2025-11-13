# ----------------------------------------------------------
# 🔹 Multi-Source Preisabruf mit Primärquelle und Fallback
# ----------------------------------------------------------
PRIMARY_SOURCE = "Yahoo Finance"
FALLBACK_USED = False
today_price = None

# Primärquelle
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

# Fallback
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
# 🔹 Historie aktualisieren
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
# 🔹 Vorheriges Ergebnis einlesen für Unterschied
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
# 🔹 Textdateien speichern
# ----------------------------------------------------------
with open("result.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("✅ Ergebnis in result.txt gespeichert.\n")
print(msg)

with open(PREVIOUS_FILE, "w", encoding="utf-8") as f:
    f.write(msg)
print("💾 previous_result.txt aktualisiert.")

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
    print(f"📄 Neue Logdatei {LOG_CSV} erstellt.")
else:
    df_log = pd.read_csv(LOG_CSV)
    df_log = pd.concat([df_log, pd.DataFrame([row])], ignore_index=True)
    df_log.to_csv(LOG_CSV, index=False, encoding="utf-8")
    print(f"📊 Ergebnis in {LOG_CSV} angehängt.")
