import time

def load_data_te():
    print("⏳ Lade Daten von TradingEconomics ...")
    url_gas = f"https://api.tradingeconomics.com/historical/commodity/natgas?c={TRADINGECONOMICS_KEY}"
    url_oil = f"https://api.tradingeconomics.com/historical/commodity/oil?c={TRADINGECONOMICS_KEY}"

    for attempt in range(1, 4):  # 3 Versuche
        r_gas = requests.get(url_gas)
        r_oil = requests.get(url_oil)

        if r_gas.status_code == 200 and r_oil.status_code == 200:
            break
        else:
            print(f"⚠️ Versuch {attempt} fehlgeschlagen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")
            if attempt < 3:
                print("⏳ Warte 5 Sekunden und versuche erneut ...")
                time.sleep(5)
            else:
                raise ValueError(f"❌ Fehler beim Abrufen nach 3 Versuchen: Gas {r_gas.status_code}, Oil {r_oil.status_code}")

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
