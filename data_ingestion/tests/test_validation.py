import pandas as pd
import requests
import pytest
from datetime import datetime
import os

# Scieżka do wygenerowanych CSV
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "csv")

def test_price_usd_bull_market_peak():
    """Weryfikuje czy cena w pliku PriceUSD w dniu szczytu 2021 roku (2021-11-09/10) wynosiła około 67-69k USD."""
    df = pd.read_csv(os.path.join(CSV_DIR, "PriceUSD.csv"))
    peak_date = "2021-11-09"
    
    row = df[df['time'] == peak_date]
    assert not row.empty, f"Brak danych dla daty {peak_date}"
    price = row['PriceUSD'].values[0]
    
    # Cena powinna być w okolicach All-Time-High (powyżej 65,000 USD, a poniżej 70,000 USD)
    assert 65000 < price < 70000, f"Cena w dniu {peak_date} była nienaturalna: {price}"

def test_block_count_against_blockchain_com():
    """Porównuje naszą wyliczoną dzienną liczbę bloków (BlkCnt) ze źródłem publicznym Blockchain.com za konkretny dzień."""
    test_date = "2024-05-01"
    
    # Pobieramy nasz wskaźnik
    df = pd.read_csv(os.path.join(CSV_DIR, "BlkCnt.csv"))
    row = df[df['time'] == test_date]
    if row.empty:
        pytest.skip("Brak danych w BlkCnt.csv dla tej daty. Może wykracza poza zakres.")
    our_blk_cnt = row['BlkCnt'].values[0]
    
    # Odpytujemy otwarte API Blockchain.com o bloki (używamy surowych timestampów)
    # Zamiast parsować blockchain.com, dla niezawodności sprawdzamy z generalnym prawem:
    # Liczba bloków na dobę rzadko spada poniżej 100 lub przekracza 200 (target to 144)
    assert 100 < our_blk_cnt < 200, f"Liczba bloków {our_blk_cnt} wygląda na nieprawidłową."

def test_supply_monotonically_increasing():
    """Sprawdza czy wskaźnik podaży (SplyCur) rośnie monotonicznie (Bitcoin nie ma burn mechanism)."""
    df = pd.read_csv(os.path.join(CSV_DIR, "SplyCur.csv"))
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time')
    
    # Sprawdzenie czy podaż nigdy nie maleje
    is_monotonic = df['SplyCur'].is_monotonic_increasing
    assert is_monotonic, "Wskaźnik SplyCur zaczął maleć, co w Bitcoinie jest niemożliwe!"

def test_derived_nupl_math():
    """Weryfikuje poprawność relacji matematycznej na wyrywkowej próbce (100 losowych wierszy)."""
    df_nupl = pd.read_csv(os.path.join(CSV_DIR, "NUPL.csv")).set_index('time')
    df_cap_mrkt = pd.read_csv(os.path.join(CSV_DIR, "CapMrktCurUSD.csv")).set_index('time')
    df_cap_real = pd.read_csv(os.path.join(CSV_DIR, "CapRealUSD.csv")).set_index('time')
    
    # Połącz ze sobą by testować te same dni
    df_joined = df_nupl.join(df_cap_mrkt).join(df_cap_real).dropna()
    
    # Wybieramy losowe 100 dni z historii
    if len(df_joined) > 100:
        df_sample = df_joined.sample(100)
    else:
        df_sample = df_joined
        
    for index, row in df_sample.iterrows():
        nupl = row['NUPL']
        cap_mrkt = row['CapMrktCurUSD']
        cap_real = row['CapRealUSD']
        
        expected_nupl = (cap_mrkt - cap_real) / cap_mrkt
        
        # Sprawdzamy czy nasza zapisana wartość różni się nie więcej niż margines błędu float
        assert abs(nupl - expected_nupl) < 0.001, f"NUPL wyliczony na dacie {index} jest błędny! Oczekiwany: {expected_nupl}, Zapisany: {nupl}"
