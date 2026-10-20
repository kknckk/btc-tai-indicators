"""Jednolity wrapper na modele fundamentalne szeregów czasowych (TSFM).

Obsługiwane modele
------------------
* ``chronos``      – Amazon Chronos (T5, ``chronos-forecasting`` v1/v2),
* ``chronos-bolt`` – Amazon Chronos-Bolt (szybszy wariant Chronos),
* ``timesfm``      – Google TimesFM (API 1.x/2.x),
* ``moirai``       – Salesforce Moirai (``uni2ts`` + ``gluonts``),
* ``stub``         – baseline random-walk z empirycznymi kwantylami (BEZ sieci
  neuronowych); jawnie oznaczony jako atrapa, używany w ``--dry-run`` oraz w API,
  gdy nie ma artefaktu modelu.

Kontrakt
--------
Wszystkie modele dziedziczą po :class:`BaseForecaster` i mają:

* ``fit(series)`` – dla TSFM to no-op (zero-shot): zapamiętuje kontekst i zwraca
  ``self``; nie ma treningu, więc nie ma ryzyka wycieku danych,
* ``predict(series=None, horizon=None) -> ForecastResult`` – prognoza dla jednej
  serii (kontekst z ``fit`` albo podany wprost),
* ``predict_batch(list_of_series, horizon=None) -> list[ForecastResult]`` –
  wsadowo, bo TSFM i tak liczą batchami (kluczowe dla rolling origin na CPU).

Wynik zawsze zawiera kwantyle ``p10``/``p50``/``p90`` (poziomy konfigurowalne).
Importy bibliotek są leniwe: brak zależności podnosi :class:`DependencyMissing`
z komunikatem „co zainstalować”, a nie surowy ``ImportError``.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

TSFM_MODELS: Tuple[str, ...] = ("chronos", "chronos-bolt", "timesfm", "moirai")
STUB_MODEL = "stub"
SUPPORTED_MODELS: Tuple[str, ...] = TSFM_MODELS + (STUB_MODEL,)

DEFAULT_MODEL_IDS: Dict[str, str] = {
    "chronos": "amazon/chronos-t5-small",
    "chronos-bolt": "amazon/chronos-bolt-small",
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "moirai": "Salesforce/moirai-1.0-R-small",
}

#: Pakiety wymagane przez poszczególne adaptery (do raportowania statusu).
MODEL_PACKAGES: Dict[str, Tuple[str, ...]] = {
    "chronos": ("torch", "chronos"),
    "chronos-bolt": ("torch", "chronos"),
    "timesfm": ("torch", "timesfm"),
    "moirai": ("torch", "uni2ts", "gluonts"),
    STUB_MODEL: (),
}

INSTALL_HINTS: Dict[str, str] = {
    "chronos": "pip install chronos-forecasting",
    "chronos-bolt": "pip install chronos-forecasting",
    "timesfm": "pip install timesfm",
    "moirai": "pip install uni2ts 'gluonts[torch]'",
}


class DependencyMissing(RuntimeError):
    """Brakuje biblioteki wymaganej przez dany model TSFM.

    Dziedziczy po ``RuntimeError``, więc istniejący kod łapiący ``Exception``
    nadal działa, ale komunikat jest czytelny dla człowieka i zawiera komendę
    instalacyjną oraz nazwę brakującego pakietu w atrybucie ``package``.
    """

    def __init__(
        self,
        model: str,
        package: str,
        install_hint: Optional[str] = None,
        detail: Optional[BaseException] = None,
    ) -> None:
        self.model = model
        self.package = package
        self.install_hint = install_hint or INSTALL_HINTS.get(model, f"pip install {package}")
        message = (
            f"Model '{model}' wymaga pakietu '{package}', którego nie ma w środowisku. "
            f"Zainstaluj: {self.install_hint}"
        )
        if detail is not None:
            message += f" (oryginalny błąd: {type(detail).__name__}: {detail})"
        super().__init__(message)


@dataclass(frozen=True)
class TSFMConfig:
    """Konfiguracja wrappera.

    Parameters
    ----------
    model:
        Jeden z :data:`SUPPORTED_MODELS`.
    context_len:
        Ile ostatnich obserwacji podajemy modelowi (przycinane od lewej).
    horizon:
        Domyślny horyzont prognozy (w krokach szeregu, tutaj: dni).
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"`` lub ``"auto"``.
    quantiles:
        Poziomy kwantyli; domyślnie 0.1/0.5/0.9.
    model_id:
        Nadpisanie identyfikatora z HuggingFace (np. większy wariant).
    batch_size:
        Rozmiar batcha przy prognozie wsadowej.
    extra:
        Dodatkowe argumenty przekazywane do konstruktora biblioteki.
    """

    model: str = "chronos"
    context_len: int = 512
    horizon: int = 7
    device: str = "cpu"
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    model_id: Optional[str] = None
    batch_size: int = 32
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model = str(self.model).strip().lower()
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Nieznany model '{self.model}'. Dostępne: {', '.join(SUPPORTED_MODELS)}"
            )
        object.__setattr__(self, "model", model)
        if int(self.context_len) < 8:
            raise ValueError("context_len musi być >= 8")
        object.__setattr__(self, "context_len", int(self.context_len))
        if int(self.horizon) < 1:
            raise ValueError("horizon musi być >= 1")
        object.__setattr__(self, "horizon", int(self.horizon))
        q = tuple(float(x) for x in self.quantiles)
        if not q or any(not 0.0 < x < 1.0 for x in q):
            raise ValueError("quantiles muszą być w przedziale (0, 1)")
        if list(q) != sorted(q):
            raise ValueError("quantiles muszą być posortowane rosnąco")
        object.__setattr__(self, "quantiles", q)
        object.__setattr__(self, "batch_size", max(1, int(self.batch_size)))

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]] = None, **overrides: Any) -> "TSFMConfig":
        data = dict(payload or {})
        data.update({k: v for k, v in overrides.items() if v is not None})
        if "quantiles" in data and data["quantiles"] is not None:
            data["quantiles"] = tuple(data["quantiles"])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_len": self.context_len,
            "horizon": self.horizon,
            "device": self.device,
            "quantiles": list(self.quantiles),
            "model_id": self.model_id or DEFAULT_MODEL_IDS.get(self.model),
            "batch_size": self.batch_size,
            "extra": dict(self.extra),
        }


def _quantile_label(level: float) -> str:
    return f"p{int(round(level * 100))}"


@dataclass
class ForecastResult:
    """Prognoza jednej serii: macierze kwantyli o długości ``horizon``."""

    model: str
    horizon: int
    quantiles: Dict[float, np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: Dict[float, np.ndarray] = {}
        for level, values in self.quantiles.items():
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size != self.horizon:
                raise ValueError(
                    f"Kwantyl {level} ma {arr.size} wartości, oczekiwano {self.horizon}"
                )
            cleaned[float(level)] = arr
        if not cleaned:
            raise ValueError("ForecastResult wymaga co najmniej jednego kwantyla")
        self.quantiles = dict(sorted(cleaned.items()))
        self.horizon = int(self.horizon)

    @property
    def levels(self) -> List[float]:
        return list(self.quantiles.keys())

    @property
    def point(self) -> np.ndarray:
        """Prognoza punktowa: mediana (p50), a bez niej środkowy kwantyl."""
        if 0.5 in self.quantiles:
            return self.quantiles[0.5]
        return self.quantiles[self.levels[len(self.levels) // 2]]

    @property
    def p10(self) -> np.ndarray:
        return self.quantiles[self.levels[0]]

    @property
    def p50(self) -> np.ndarray:
        return self.point

    @property
    def p90(self) -> np.ndarray:
        return self.quantiles[self.levels[-1]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "horizon": self.horizon,
            "point_forecast": [float(x) for x in self.point],
            "quantiles": {
                _quantile_label(level): [float(x) for x in values]
                for level, values in self.quantiles.items()
            },
            "meta": self.meta,
        }


def _as_clean_array(series: Sequence[float], name: str = "series") -> np.ndarray:
    """Konwersja do 1-D float, interpolacja NaN, walidacja długości."""
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} jest pusta")
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise ValueError(f"{name} ma mniej niż 2 skończone obserwacje")
    if not finite.all():
        idx = np.arange(arr.size)
        arr = np.interp(idx, idx[finite], arr[finite])
    return arr


def _coerce_quantile_array(
    raw: Any,
    batch_size: int,
    horizon: int,
    n_levels: int,
    model: str,
) -> np.ndarray:
    """Sprowadza wynik biblioteki do kształtu ``(batch, horizon, n_levels)``.

    Chronos v1 zwraca listę tensorów ``(n_levels, horizon)``, Chronos v2/Bolt
    tensor ``(batch, horizon, n_levels)`` – oba warianty są tu normalizowane.
    """
    if isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            if hasattr(item, "detach"):
                parts.append(item.detach().cpu().numpy())
            else:
                parts.append(np.asarray(item, dtype=float))
        arr = np.stack(parts, axis=0)
    elif hasattr(raw, "detach"):
        arr = raw.detach().cpu().numpy()
    else:
        arr = np.asarray(raw, dtype=float)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"{model}: nieoczekiwany kształt kwantyli {arr.shape}")

    if arr.shape[1] == n_levels and arr.shape[2] == horizon:
        arr = np.transpose(arr, (0, 2, 1))
    if arr.shape[2] != n_levels:
        raise RuntimeError(
            f"{model}: oczekiwano {n_levels} kwantyli, otrzymano kształt {arr.shape}"
        )
    if arr.shape[1] != horizon:
        raise RuntimeError(
            f"{model}: oczekiwano horyzontu {horizon}, otrzymano kształt {arr.shape}"
        )
    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)
    if arr.shape[0] != batch_size:
        raise RuntimeError(
            f"{model}: oczekiwano batcha {batch_size}, otrzymano {arr.shape[0]}"
        )
    return arr


class BaseForecaster(ABC):
    """Wspólny interfejs modeli TSFM (zero-shot)."""

    name: str = "base"
    is_stub: bool = False

    def __init__(self, config: Optional[TSFMConfig] = None, **overrides: Any) -> None:
        if config is None:
            config = TSFMConfig(model=self.name, **overrides)  # type: ignore[arg-type]
        elif overrides:
            payload = config.to_dict()
            payload.update(overrides)
            config = TSFMConfig.from_dict(payload)
        self.config = config
        self._context: Optional[np.ndarray] = None

    # -- API publiczne ---------------------------------------------------
    def fit(self, series: Sequence[float]) -> "BaseForecaster":
        """Zero-shot: zapamiętuje kontekst, nie trenuje parametrów."""
        self._context = _as_clean_array(series)
        return self

    def predict(
        self,
        series: Optional[Sequence[float]] = None,
        horizon: Optional[int] = None,
    ) -> ForecastResult:
        if series is None and self._context is None:
            raise ValueError("Brak kontekstu: wywołaj fit(series) albo podaj series=...")
        context = self._context if series is None else _as_clean_array(series)
        if context is None:  # pragma: no cover - zabezpieczenie dla type checkerów
            raise ValueError("Brak kontekstu: wywołaj fit(series) albo podaj series=...")
        return self.predict_batch([context], horizon=horizon)[0]

    def predict_batch(
        self,
        series_list: Sequence[Sequence[float]],
        horizon: Optional[int] = None,
    ) -> List[ForecastResult]:
        h = int(horizon or self.config.horizon)
        if h < 1:
            raise ValueError("horizon musi być >= 1")
        if not series_list:
            return []

        contexts: List[np.ndarray] = []
        for i, series in enumerate(series_list):
            arr = _as_clean_array(series, name=f"series_list[{i}]")
            if arr.size < 8:
                raise ValueError(f"series_list[{i}] ma {arr.size} obserwacji, minimum to 8")
            contexts.append(arr[-self.config.context_len :])

        quantiles = self._predict_batch(contexts, h)
        quantiles = np.sort(np.asarray(quantiles, dtype=float), axis=2)  # monotoniczność
        results: List[ForecastResult] = []
        meta = self.describe()
        for row in quantiles:
            results.append(
                ForecastResult(
                    model=self.config.model,
                    horizon=h,
                    quantiles={
                        level: row[:, idx]
                        for idx, level in enumerate(self.config.quantiles)
                    },
                    meta=dict(meta),
                )
            )
        return results

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "model_id": self.config.model_id or DEFAULT_MODEL_IDS.get(self.config.model),
            "context_len": self.config.context_len,
            "device": self.config.device,
            "stub": self.is_stub,
            "quantiles": list(self.config.quantiles),
        }

    # -- API dla podklas --------------------------------------------------
    @abstractmethod
    def _predict_batch(self, contexts: List[np.ndarray], horizon: int) -> np.ndarray:
        """Zwraca tablicę ``(batch, horizon, len(config.quantiles))``."""


class StubForecaster(BaseForecaster):
    """Baseline random-walk z empirycznymi kwantylami (bez sieci neuronowej).

    Prognoza punktowa = ostatnia cena (błądzenie losowe bez dryfu), a pasmo
    p10/p90 to ``exp(±1.2816 * sigma_d * sqrt(h))``, gdzie ``sigma_d`` to
    odchylenie standardowe dziennych log-zwrotów z ostatnich 90 obserwacji.

    To **nie jest** model TSFM – służy do testowania całego pipeline'u bez
    ciężkich zależności (``--dry-run`` w ``experiments/run_tsfm.py``) oraz jako
    jawny fallback API, gdy brak artefaktu. W wynikach zawsze ma
    ``meta["stub"] = True``.
    """

    name = STUB_MODEL
    is_stub = True

    def _predict_batch(self, contexts: List[np.ndarray], horizon: int) -> np.ndarray:
        from scipy.stats import norm  # scipy jest już zależnością repo

        # Parametry mogą pochodzić z artefaktu (``extra``), inaczej są estymowane
        # z ostatnich 90 log-zwrotów – dzięki temu ten sam baseline obsługuje
        # zarówno --dry-run, jak i "artefakt" API po offline'owym ficie.
        sigma_override = self.config.extra.get("sigma_daily")
        drift_override = float(self.config.extra.get("drift_daily", 0.0) or 0.0)

        out = np.empty((len(contexts), horizon, len(self.config.quantiles)), dtype=float)
        levels = self.config.quantiles
        for i, context in enumerate(contexts):
            if sigma_override is not None:
                sigma = max(float(sigma_override), 1e-6)
            else:
                window = min(90, max(2, context.size - 1))
                log_returns = np.diff(np.log(np.clip(context[-window - 1 :], 1e-12, None)))
                sigma = float(np.std(log_returns, ddof=1)) if log_returns.size > 1 else 0.0
                sigma = max(sigma, 1e-6)
            last = float(context[-1])
            steps = np.arange(1, horizon + 1, dtype=float)
            median = last * np.exp(drift_override * steps)
            # kwantyle log-normalne wokół mediany
            for idx, level in enumerate(levels):
                z = float(norm.ppf(level))
                out[i, :, idx] = median * np.exp(z * sigma * np.sqrt(steps))
        return out

    def describe(self) -> Dict[str, Any]:
        meta = super().describe()
        meta.update(
            {
                "method": "random_walk_empirical_quantiles",
                "sigma_window": 90,
                "sigma_daily": self.config.extra.get("sigma_daily"),
                "drift_daily": self.config.extra.get("drift_daily", 0.0),
                "note": "ATRAPA (stub) – nie jest to wytrenowany model TSFM",
            }
        )
        return meta


class ChronosForecaster(BaseForecaster):
    """Adapter Amazon Chronos / Chronos-Bolt (``chronos-forecasting``)."""

    name = "chronos"

    def __init__(self, config: Optional[TSFMConfig] = None, **overrides: Any) -> None:
        super().__init__(config=config, **overrides)
        self._pipeline: Any = None

    # -- ładowanie modelu ------------------------------------------------
    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - zależy od środowiska
            raise DependencyMissing(self.config.model, "torch", "pip install torch", exc) from exc
        try:
            import chronos  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissing(
                self.config.model, "chronos-forecasting", INSTALL_HINTS["chronos"], exc
            ) from exc

        if self.config.model == "chronos-bolt":
            cls = getattr(chronos, "ChronosBoltPipeline", None) or getattr(
                chronos, "BaseChronosPipeline", None
            )
        else:
            cls = getattr(chronos, "ChronosPipeline", None) or getattr(
                chronos, "BaseChronosPipeline", None
            )
        if cls is None:
            raise DependencyMissing(
                self.config.model,
                "chronos-forecasting>=1.4",
                INSTALL_HINTS["chronos"],
                detail=RuntimeError("zainstalowana wersja nie eksportuje ChronosPipeline"),
            )

        model_id = self.config.model_id or DEFAULT_MODEL_IDS[self.config.model]
        device = "cpu" if self.config.device in ("auto", "", None) else self.config.device
        kwargs: Dict[str, Any] = {"device_map": device}
        if torch.cuda.is_available() and device.startswith("cuda"):
            kwargs["torch_dtype"] = torch.bfloat16
        kwargs.update(self.config.extra)
        try:
            self._pipeline = cls.from_pretrained(model_id, **kwargs)
        except TypeError as exc:  # starsze API bez torch_dtype
            LOGGER.warning("Chronos: from_pretrained odrzucił %s (%s) – próba bez torch_dtype", kwargs, exc)
            kwargs.pop("torch_dtype", None)
            self._pipeline = cls.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - sieć/dysk
            raise RuntimeError(
                f"Chronos: nie udało się załadować '{model_id}' na '{device}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return self._pipeline

    def _predict_batch(self, contexts: List[np.ndarray], horizon: int) -> np.ndarray:
        import torch

        pipeline = self._load()
        tensor = torch.stack(
            [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts], dim=0
        )
        device = getattr(pipeline, "device", None)
        if device is not None:
            tensor = tensor.to(device)
        try:
            raw = pipeline.predict_quantiles(
                inputs=tensor,
                prediction_length=horizon,
                quantile_levels=list(self.config.quantiles),
            )
        except TypeError:
            # API v1 używa nazwy ``context`` zamiast ``inputs``
            try:
                raw = pipeline.predict_quantiles(
                    context=[torch.tensor(ctx, dtype=torch.float32) for ctx in contexts],
                    prediction_length=horizon,
                    quantile_levels=list(self.config.quantiles),
                )
            except TypeError:
                raw = pipeline.predict_quantiles(
                    [torch.tensor(ctx, dtype=torch.float32) for ctx in contexts],
                    horizon,
                    list(self.config.quantiles),
                )
        if isinstance(raw, tuple):  # (quantiles, mean) w chronos>=2.0
            raw = raw[0]
        return _coerce_quantile_array(
            raw, len(contexts), horizon, len(self.config.quantiles), self.config.model
        )


def _select_quantile_levels(q: np.ndarray, levels: Sequence[float], model: str) -> np.ndarray:
    """Wybiera żądane poziomy z pełnej siatki kwantyli zwróconej przez model.

    TimesFM zwraca 9-10 decyli (0.1..0.9); wybieramy najbliższy indeks dla
    każdy żądany poziom. Gdy model zwrócił dokładnie tyle kwantyli, ile
    żądano – zwracamy bez zmian.
    """
    n_q = q.shape[2]
    if n_q == len(levels):
        return q
    if n_q == 3:
        return q[:, :, [0, 1, 2]]
    grid = np.linspace(0.1, 0.9, n_q)
    idx = [int(np.argmin(np.abs(grid - level))) for level in levels]
    if len(set(idx)) != len(idx):
        raise RuntimeError(
            f"{model}: nie da się wybrać poziomów {list(levels)} z siatki {n_q} kwantyli"
        )
    return q[:, :, idx]


class TimesFMForecaster(BaseForecaster):
    """Adapter Google TimesFM (API 2.5/3.x oraz legacy 1.x; import leniwy).

    Preferowane API to ``TimesFM_2p5_200M_torch`` (``timesfm>=2.5``):
    ``from_pretrained`` → ``compile(ForecastConfig(...))`` → ``forecast``.
    W starszych wersjach używany jest ``TimesFm`` + ``TimesFmHparams``.
    """

    name = "timesfm"

    def __init__(self, config: Optional[TSFMConfig] = None, **overrides: Any) -> None:
        super().__init__(config=config, **overrides)
        self._model: Any = None
        self._api: Optional[str] = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissing("timesfm", "torch", "pip install torch", exc) from exc
        try:
            import timesfm
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissing("timesfm", "timesfm", INSTALL_HINTS["timesfm"], exc) from exc

        model_id = self.config.model_id or DEFAULT_MODEL_IDS["timesfm"]
        device = "cpu" if self.config.device in ("auto", "", None) else self.config.device

        # --- API 2.5+: TimesFM_2p5_200M_torch ---------------------------
        cls = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if cls is not None:
            model_id = self.config.model_id or "google/timesfm-2.5-200m-pytorch"
            try:
                try:
                    model = cls.from_pretrained(model_id, torch_compile=False)
                except TypeError:
                    model = cls.from_pretrained(model_id)
                forecast_config = timesfm.ForecastConfig(
                    max_context=self.config.context_len,
                    max_horizon=max(self.config.horizon, int(self.config.extra.get("max_horizon", 0))),
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                    per_core_batch_size=self.config.batch_size,
                )
                if hasattr(model, "to"):
                    model = model.to(device)
                model.compile(forecast_config)
            except Exception as exc:  # pragma: no cover - sieć/waga modelu
                raise RuntimeError(
                    f"TimesFM: nie udało się załadować '{model_id}' (API 2.5): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._model, self._api = model, "2.5"
            return self._model

        # --- API 1.x: TimesFmHparams/TimesFmCheckpoint ------------------
        if hasattr(timesfm, "TimesFmHparams") and hasattr(timesfm, "TimesFm"):
            model_id = self.config.model_id or "google/timesfm-1.0-200m-pytorch"
            try:
                hparams = timesfm.TimesFmHparams(
                    backend=device,
                    per_core_batch_size=self.config.batch_size,
                    horizon=self.config.horizon,
                )
                checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id)
                self._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"TimesFM: nie udało się załadować '{model_id}' (API 1.x): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._api = "1.x"
            return self._model

        raise RuntimeError(
            "TimesFM: zainstalowana wersja nie udostępnia rozpoznanego API "
            "(TimesFM_2p5_200M_torch ani TimesFm) – sprawdź wersję pakietu."
        )

    def _predict_batch(self, contexts: List[np.ndarray], horizon: int) -> np.ndarray:
        model = self._load()
        inputs = [ctx.astype(np.float32) for ctx in contexts]
        try:
            if self._api == "2.5":
                _point, quantiles = model.forecast(horizon=horizon, inputs=inputs)
            else:  # legacy 1.x
                _point, quantiles = model.forecast(inputs=inputs, freq=[0] * len(contexts))
        except Exception as exc:  # pragma: no cover - API zależne od wersji
            raise RuntimeError(
                f"TimesFM: prognoza nie powiodła się ({type(exc).__name__}: {exc})"
            ) from exc

        q = np.asarray(quantiles, dtype=float)
        if q.ndim != 3:
            raise RuntimeError(f"TimesFM: nieoczekiwany kształt kwantyli {q.shape}")
        q = _select_quantile_levels(q, self.config.quantiles, "timesfm")
        return _coerce_quantile_array(
            q, len(contexts), horizon, len(self.config.quantiles), "timesfm"
        )


class MoiraiForecaster(BaseForecaster):
    """Adapter Salesforce Moirai przez ``uni2ts`` + ``gluonts``.

    Ścieżka najlepszego wysiłku: nie była uruchamiana w tym środowisku
    (brak ``uni2ts`` i ``gluonts``). Brak zależności daje ``DependencyMissing``
    z komendą instalacyjną, a nie traceback importu.
    """

    name = "moirai"

    def __init__(self, config: Optional[TSFMConfig] = None, **overrides: Any) -> None:
        super().__init__(config=config, **overrides)
        self._predictor: Any = None

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        try:
            import torch  # noqa: F401
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissing(
                "moirai", "uni2ts", "pip install uni2ts 'gluonts[torch]'", exc
            ) from exc
        try:
            from gluonts.model.predictor import Predictor  # noqa: F401
            from gluonts.torch import PyTorchPredictor
        except ImportError as exc:  # pragma: no cover
            raise DependencyMissing(
                "moirai", "gluonts", "pip install 'gluonts[torch]'", exc
            ) from exc

        model_id = self.config.model_id or DEFAULT_MODEL_IDS["moirai"]
        try:
            module = MoiraiModule.from_pretrained(model_id)
            forecast = MoiraiForecast(
                module=module,
                prediction_length=self.config.horizon,
                context_length=self.config.context_len,
                patch_size="auto",
                num_samples=self.config.extra.get("num_samples", 100),
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self._predictor = PyTorchPredictor(
                prediction_length=self.config.horizon,
                input_names=["target"],
                prediction_net=forecast,
                batch_size=self.config.batch_size,
                device=self.config.device,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Moirai: nie udało się zbudować predyktora dla '{model_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return self._predictor

    def _predict_batch(self, contexts: List[np.ndarray], horizon: int) -> np.ndarray:
        # Najpierw walidacja zależności (_load rzuca DependencyMissing), dopiero
        # potem import gluonts – inaczej brak pakietu wycieka jako surowy ImportError.
        predictor = self._load()
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        entries = []
        for ctx in contexts:
            entries.append(
                {
                    "start": pd.Period("2000-01-01", freq="D"),
                    "target": ctx.astype(np.float32),
                }
            )
        dataset = ListDataset(entries, freq="D")
        forecasts = list(predictor.predict(dataset))
        out = np.empty((len(contexts), horizon, len(self.config.quantiles)), dtype=float)
        for i, fc in enumerate(forecasts):
            for j, level in enumerate(self.config.quantiles):
                out[i, :, j] = np.asarray(fc.quantile(level), dtype=float)[:horizon]
        return out


FORECASTERS: Dict[str, type] = {
    "chronos": ChronosForecaster,
    "chronos-bolt": ChronosForecaster,
    "timesfm": TimesFMForecaster,
    "moirai": MoiraiForecaster,
    STUB_MODEL: StubForecaster,
}


def create_forecaster(
    config: Optional[TSFMConfig] = None,
    model: Optional[str] = None,
    **overrides: Any,
) -> BaseForecaster:
    """Fabryka: ``create_forecaster(model="chronos-bolt", horizon=7)``."""
    payload: Dict[str, Any] = dict(config.to_dict()) if config is not None else {}
    if model is not None:
        payload["model"] = model
    payload.update({k: v for k, v in overrides.items() if v is not None})
    cfg = TSFMConfig.from_dict(payload)
    forecaster_cls = FORECASTERS.get(cfg.model)
    if forecaster_cls is None:  # pragma: no cover - walidacja w TSFMConfig
        raise ValueError(f"Brak adaptera dla modelu '{cfg.model}'")
    return forecaster_cls(cfg)


def zero_shot_forecast(
    series: Sequence[float],
    horizon: int = 7,
    model: str = "chronos",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Prognoza zero-shot jednym wywołaniem, wynik gotowy do JSON.

    Przykład
    --------
    >>> zero_shot_forecast(prices, h=7, model="chronos-bolt")["quantiles"].keys()
    dict_keys(['p10', 'p50', 'p90'])
    """
    forecaster = create_forecaster(model=model, horizon=horizon, **kwargs)
    result = forecaster.fit(series).predict(horizon=horizon)
    return result.to_dict()


def _module_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def available_models() -> List[Dict[str, Any]]:
    """Status modeli bez ich importowania (bezpieczne dla API/CLI)."""
    status: List[Dict[str, Any]] = []
    for model in SUPPORTED_MODELS:
        packages = MODEL_PACKAGES[model]
        missing = [pkg for pkg in packages if not _module_installed(pkg)]
        status.append(
            {
                "model": model,
                "available": not missing,
                "missing_packages": missing,
                "install_hint": INSTALL_HINTS.get(model),
                "stub": model == STUB_MODEL,
            }
        )
    return status


__all__ = [
    "DependencyMissing",
    "TSFMConfig",
    "ForecastResult",
    "BaseForecaster",
    "StubForecaster",
    "ChronosForecaster",
    "TimesFMForecaster",
    "MoiraiForecaster",
    "create_forecaster",
    "zero_shot_forecast",
    "available_models",
    "SUPPORTED_MODELS",
    "TSFM_MODELS",
    "DEFAULT_MODEL_IDS",
]
