# ----------------------------------------------------------
# 🔍 TradingEconomics Auto-Slug Finder + Update Script
# ----------------------------------------------------------
import requests
import re
import os

# === Dein API-Key ===
TRADINGECONOMICS_KEY = "DEIN_KEY_HIER"  # <-- hier deinen echten Key eintragen
TARGET_FILE = "gas_forecast_daily.py"

# === Kandidaten für Commodities ===
GAS_VARIANTS = ["NG", "NaturalGas", "Natural-Gas", "Natural Gas", "naturalgas", "natgas"]
OIL_VARIANTS = ["CrudeOil", "Crude-Oil", "Crude Oil", "OIL", "Brent", "WTI"]

def test_variant(commodity, variants):
    """Teste Varianten und gib die erste funktionierende zurück."""
    for var in variants:
        url = f"https://api.tradingeconomics.com/commodity/{var}?c={TRADINGECONOMICS_KEY}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                print(f"✅ {commodity}: Erfolgreich mit '{var}' (Status 200)")
                return var
            else:
                print(f"⚠️ {commodity}: {var} → Status {r.status_code}")
        except Exception as e:
            print(f"❌ {commodity}: Fehler bei {var} → {e}")
    return None

def update_file(symbol_gas, symbol_oil):
    """Aktualisiert die SYMBOL_* Variablen in gas_forecast_daily.py"""
    if not os.path.exists(TARGET_FILE):
        print(f"❌ Datei {TARGET_FILE} nicht gefunden!")
        return False

    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'SYMBOL_GAS\s*=\s*".*"', f'SYMBOL_GAS = "{symbol_gas}"', content)
    content = re.sub(r'SYMBOL_OIL\s*=\s*".*"', f'SYMBOL_OIL = "{symbol_oil}"', content)

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"💾 {TARGET_FILE} wurde mit neuen Symbolen aktualisiert:")
    print(f"   SYMBOL_GAS = {symbol_gas}")
    print(f"   SYMBOL_OIL = {symbol_oil}")
    return True

if __name__ == "__main__":
    print("🔍 Teste TradingEconomics Commodity Endpoints...\n")

    gas_slug = test_variant("Natural Gas", GAS_VARIANTS)
    oil_slug = test_variant("Crude Oil", OIL_VARIANTS)

    if gas_slug and oil_slug:
        update_file(gas_slug, oil_slug)
        print("✅ Beide Slugs erfolgreich aktualisiert.")
    else:
        print("⚠️ Nicht alle Slugs konnten ermittelt werden.")
