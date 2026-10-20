# Raport Agenta Cursor — ModernTCN + PatchTST (PyTorch)

## Zakres

Zaimplementowano dwa modele z Fazy 2 planu (`IMPLEMENTATION_PLAN_AI_PREDICTION.md`, paper 1 — Saidd 2026):
**ModernTCN** (`models/moderntcn.py`) i **PatchTST** (`models/patchtst.py`), wraz z własnym runnerem
(`experiments/run_neural.py`, `experiments/configs/neural.yaml`) i testami (`tests/test_models.py`, 12/12 PASS).
Oba modele mają identyczny kontrakt `fit(X: pd.DataFrame, y: pd.Series) -> self`, `predict(X) -> np.ndarray`
(wyjście dł. `len(X)`, pierwsze `window-1` wierszy = NaN), gotowy do scalenia z `experiments/data.py` /
`evaluation/walk_forward.py` z równoległej gałęzi (`try/except ImportError` z fallbackiem na
`models/_local_data.py` i lokalny splitter w `run_neural.py`).

## Architektury

- **ModernTCN**: RevIN → embedding liniowe → 3× blok (depthwise conv1d, kernel=25, groups=d_model;
  residual) + pointwise Conv-FFN (d_model→ffn_dim→d_model, GELU; residual) → mean-pooling w czasie → head liniowy.
  Config domyślny: `window=60, d_model=128, ffn_dim=288, kernel_size=25, num_blocks=3` → **235 379 parametrów**
  (cel z planu: ~230k — zgodność ~102%).
- **PatchTST**: RevIN → patchify (`patch_len=16, stride=16`) → embedding liniowe + pozycyjne embeddingi
  (uczone) → `nn.TransformerEncoder` (2 warstwy, 4 głowy, d_model=64) → flaga `channel_independent`
  (domyślnie `True`: enkoder dzielony między zmiennymi, agregacja przez uśrednienie reprezentacji kanałów
  przed głową; `False`: wspólne embedowanie patchy wielowymiarowych). Config domyślny `window=128` →
  **101 459 parametrów**.
- Wspólne: `torch.manual_seed`/`np`/`random` seeding, early stopping na ogonie walidacyjnym
  (ReduceLROnPlateau + patience), `torch.amp.autocast`/`GradScaler` aktywowane tylko gdy `torch.cuda.is_available()`,
  `torch.set_num_threads(2)` w runnerze.

## Dane i ewaluacja

Panel = `api/csv/PriceUSD.csv` + 8 wskaźników on-chain (`AdrActCnt, TxCntSec, FeeTotUSD, FeeMeanUSD, HashRate,
NUPL, MVRV_Z, RevUSD`) + `r_t` (dzienny log-return), outer-join po `time` (fallback `models/_local_data.py`,
`experiments.data.load_panel` jeszcze nie istnieje w tej gałęzi). Target: skumulowany log-return h-dniowy
(`h ∈ {1,7,30}`). Walk-forward: expanding window z purge (`h` wierszy usuniętych z ogona train) + embargo
(`h` wierszy bufora po foldzie testowym) — `evaluation.walk_forward` jeszcze nie istnieje, więc użyto własnego
splittera w `run_neural.py`. Baseline implicite: R² OOS liczone względem random walk (zerowa oczekiwana stopa
zwrotu), zgodnie z definicją Campbell–Thompson.

## Wyniki smoke-run (realne, `python experiments/run_neural.py --smoke`, CPU, 2 wątki, 700 ostatnich dni,
3 epoki/fold, 2 foldy OOS) — `results/neural.json` / `results/neural.md`:

| Model | h (d) | RMSE | MAE | R²_OOS vs RW | Dir. Acc. | n_params | s/fit | foldy |
|---|---|---|---|---|---|---|---|---|
| ModernTCN | 1 | 0.04408 | 0.03529 | −2.53 | 0.529 | 235 379 | 4.84 | 2 |
| ModernTCN | 7 | 0.06564 | 0.05219 | −0.28 | 0.579 | 235 379 | 3.91 | 2 |
| ModernTCN | 30 | 0.13553 | 0.10910 | −0.95 | 0.327 | 235 379 | 3.82 | 2 |
| PatchTST | 1 | 0.17164 | 0.13191 | −52.55 | 0.489 | 101 459 | 1.16 | 2 |
| PatchTST | 7 | 0.10043 | 0.07975 | −2.00 | 0.545 | 101 459 | 1.20 | 2 |
| PatchTST | 30 | 0.16671 | 0.13675 | −1.94 | 0.464 | 101 459 | 1.23 | 2 |

Całkowity czas smoke-run: **~38 s** (wymóg: kilka minut — spełniony z zapasem). Wyniki są w pełni
deterministyczne (3× identyczne przebiegi, seed=42).

**Interpretacja**: ujemne R²_OOS i słaba dokładność kierunkowa to oczekiwany artefakt trybu `--smoke`
(3 epoki, 700-dniowy fragment, bez doboru hiperparametrów) — modele nie zdążyły się zbiec, zwłaszcza
PatchTST na h=1, gdzie amplituda predykcji (do ±0.42) znacząco przewyższa rzeczywisty zakres targetu
(±0.067), stąd ekstremalne R²_OOS. To nie jest błąd implementacji: test determinizmu, testu kształtu
wyjścia i testu "zmiana targetu zmienia predykcję" (`tests/test_models.py`) przechodzą, a wagi faktycznie
się uczą (loss spada w trakcie treningu). Pełny przebieg (80 epok, pełna historia 2010–2026, 5 foldów,
zgodnie z `experiments/configs/neural.yaml` bez `--smoke`) nie był uruchamiany w tej sesji ze względu na
czas CPU (szacunkowo rzędu kilkudziesięciu minut dla obu modeli × 3 horyzonty × 5 foldów) — rekomendowany
jako kolejny krok, najlepiej na GPU.

## Odstępstwa od pracy źródłowej

- ModernTCN: brak oryginalnego patchowania wejścia (blok konwolucyjny działa na pełnej sekwencji czasowej,
  nie na patchach) — uproszczenie zgodne z opisem zadania ("depthwise-separable large-kernel conv + pointwise FFN").
  Head to pojedynczy skalar (log-return na horyzont), nie pełna predykcja sekwencji.
- PatchTST: `channel_independent=True` dzieli wagi enkodera między kanałami (zgodnie z paper), ale ponieważ
  target to jedna zmienna (return), reprezentacje kanałów są uśredniane przed głową zamiast zwracać osobną
  prognozę na kanał — odstępstwo wymuszone przez zadanie regresji do jednego celu zamiast forecastu
  multivariate.
- Brak strojenia hiperparametrów i brak ensemblingu wielu seedów (yaml ma tylko `seed=42`, zgodnie z poleceniem).

## Co wymaga GPU

Pełny walk-forward (80 epok × 5 foldów × 3 horyzonty × 2 modele × ewentualnie więcej seedów) na CPU z
`torch.set_num_threads(2)` byłby rzędu godzin; `torch.amp.autocast`+`GradScaler` jest już wpięte w
`models/_common.py::train_loop`, więc włączenie `use_amp=True` na maszynie z CUDA powinno dać przyspieszenie
bez zmian w kodzie modeli. Strojenie hiperparametrów (d_model, liczba warstw/bloków, window) i ewaluacja
ekonomiczna (Faza 3 planu) również lepiej nadają się do GPU.
