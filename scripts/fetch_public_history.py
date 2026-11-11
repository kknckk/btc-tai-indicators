"""Bootstrap pelnej historii dla publicznych serii multiasset.

Dlaczego ten skrypt istnieje: `multiasset.fetch_series()` przy pustym cache pobiera
tylko ogonek (NBP: endpoint bez zakresu, Yahoo: `range=max` potrafi zwrocic ulamek
historii). Do walk-forward potrzebujemy pelnej historii:
  - NBP (EUR/PLN, USD/PLN): paginacja po ~90 dni (limit API) od 2002-01-01,
  - Yahoo (wig20, wheat, oil, gold): jawne period1=2000-01-01..dzis.

Wynik laduje w data/multiasset/<seria>.{csv,parquet} przez multiasset._write_cache.
Ucieczka/siec: kazda pajeczyna zapisuje to co juz ma; bledy chunkow sa zbierane,
nie zabijaja innych serii.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, ".")
from experiments import multiasset as ma

NBP_CODES = {"eurpln": "eur", "usdpln": "usd"}
YAHOO = {"wig20": "WIG20.WA", "wheat": "ZW=F", "oil": "CL=F", "gold": "GC=F"}
NBP_FROM = date(2002, 1, 1)
YF_FROM = date(2000, 1, 1)


def fetch_nbp_full(code: str) -> pd.Series:
    """Paginacja zakresami ~90 dni (limit NBP), laczenie i czyszczenie."""
    chunks = []
    cur = NBP_FROM
    today = date.today()
    while cur <= today:
        end = min(cur + timedelta(days=89), today)
        try:
            ch = None
            last_err = None
            for _ in range(3):  # retry per chunk — brak cichej luki
                try:
                    ch = ma._fetch_nbp(code, start=cur, end=end)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(1.5)
            if ch is None:
                raise RuntimeError(f"NBP {code} {cur}..{end}: 3x fail ({last_err})")
            chunks.append(ch)
        except Exception as e:
            print(f"  NBP {code} {cur}..{end}: FAIL {str(e)[:80]}")
            raise
        cur = end + timedelta(days=1)
        time.sleep(0.2)  # grzecznosc wobec API
    if not chunks:
        raise RuntimeError(f"NBP {code}: zero chunkow")
    s = pd.concat(chunks).sort_index()
    s.name = code
    return ma._clean_series(s)


def fetch_yahoo_full(symbol: str) -> pd.Series:
    s = ma._fetch_yahoo(symbol, start=YF_FROM, end=date.today())
    return s


def fetch_wig20_biznesradar() -> pd.Series:
    """WIG20 dziennie z biznesradar.pl (Yahoo i stooq odpadly: YF ma 1 obs,
    stooq blokuje DC po PoW). Tabela 50 wierszy/strona, paginacja notowania-historyczne/WIG20,<n>.
    Zapisuje zamkniecie (Zamkniecie) jako serie dzienna."""
    import re
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    rows: list[tuple[pd.Timestamp, float]] = []
    page, max_pages, consec_http_err = 1, 400, 0
    while page <= max_pages:
        url = f"https://www.biznesradar.pl/notowania-historyczne/WIG20,{page}"
        resp = s.get(url, timeout=30)
        if resp.status_code != 200:
            consec_http_err += 1
            if consec_http_err >= 3:
                raise RuntimeError(
                    f"biznesradar HTTP {resp.status_code} x{consec_http_err} przy page={page} — odmowa przerwania z niekompletnym cache"
                )
            time.sleep(2.0)
            continue
        consec_http_err = 0
        t = resp.text
        tr = re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S)
        got = 0
        for row in tr:
            cells = [c.strip() for c in re.findall(r"<t[dh][^>]*>([^<]*)</t[dh]>", row)]
            if len(cells) >= 5 and re.match(r"\d{2}\.\d{2}\.\d{4}", cells[0]):
                dt = pd.to_datetime(cells[0], format="%d.%m.%Y")
                close = float(cells[4].replace(" ", "").replace(",", "."))
                rows.append((dt, close))
                got += 1
        if got == 0:  # status 200 bez wierszy tabeli = rzeczywisty koniec paginacji
            break
        page += 1
        time.sleep(0.3)
    if not rows:
        raise RuntimeError("biznesradar: zero wierszy WIG20")
    idx, vals = zip(*sorted(rows))
    ser = pd.Series(vals, index=pd.DatetimeIndex(idx), name="wig20")
    return ma._clean_series(ser)


def main() -> int:
    summary = []
    for name, code in NBP_CODES.items():
        try:
            s = fetch_nbp_full(code)
            s.name = name
            ma._write_cache(name, s)
            summary.append((name, len(s), str(s.index.min().date()), str(s.index.max().date())))
        except Exception as e:
            summary.append((name, f"FAIL {str(e)[:80]}", "", ""))
    for name, sym in YAHOO.items():
        try:
            s = fetch_wig20_biznesradar() if name == "wig20" else fetch_yahoo_full(sym)
            s.name = name
            ma._write_cache(name, s)
            summary.append((name, len(s), str(s.index.min().date()), str(s.index.max().date())))
        except Exception as e:
            summary.append((name, f"FAIL {str(e)[:80]}", "", ""))
    for row in summary:
        print(" | ".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
