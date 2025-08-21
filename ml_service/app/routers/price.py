from fastapi import APIRouter
from ..schemas import PredictPriceIn, PredictPriceOut
from ..config import settings
from ..utils import timeboxed

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import numpy as np
import re

log = logging.getLogger("ml.price")
router = APIRouter(tags=["Price"])

# -------- Heuristic interval fallbacks (used if no calibration found) --------
_INTERVAL_MIN = 100.0     # PKR; minimum half-width
_INTERVAL_PCT = 0.15      # ±15% around point estimate

# -------- Globals (lazy-loaded) --------
_model: Any = None
_calib_df: Optional[pd.DataFrame] = None


# =========================
# Loading artifacts
# =========================
def _load_calibration() -> Optional[pd.DataFrame]:
    """
    Try to load per-(category, condition) absolute-error quantiles (q50/q75/q90/q95).
    Looks for settings.PRICE_CALIB_PATH or a sibling CSV near the model.
    """
    # 1) explicit path
    calib_path = getattr(settings, "PRICE_CALIB_PATH", None)
    if calib_path:
        p = Path(calib_path)
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["category"] = df["category"].astype(str)
                df["condition"] = df["condition"].astype(str)
                return df
            except Exception:
                log.exception("Failed reading calibration file at PRICE_CALIB_PATH")

    # 2) sibling CSV next to model
    try:
        model_path = Path(settings.PRICE_MODEL_PATH)
        sib = model_path.parent / "error_bands_by_cat_cond.csv"
        if sib.exists():
            df = pd.read_csv(sib)
            df["category"] = df["category"].astype(str)
            df["condition"] = df["condition"].astype(str)
            return df
    except Exception:
        log.exception("Failed reading sibling calibration CSV")

    return None


def load_model():
    global _model, _calib_df
    if _model is None:
        _model = joblib.load(settings.PRICE_MODEL_PATH)
        log.info("Loaded price model (compat-4): %s", type(_model))
        _calib_df = _load_calibration()
        if _calib_df is not None:
            log.info("Loaded calibration bands: %d rows", len(_calib_df))
        else:
            log.warning("No calibration bands found; falling back to ±%d%%/min %.0f PKR",
                        int(_INTERVAL_PCT * 100), _INTERVAL_MIN)
    return _model


# =========================
# Helpers
# =========================
def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _build_compat_row(payload: PredictPriceIn) -> pd.DataFrame:
    """
    Build the single-row DataFrame that the compat-4 pipeline expects:
      columns -> ["category", "condition", "text_concat"]
    """
    # Prefer normalized text if present; else normalize raw
    t = payload.title_norm or payload.title or ""
    d = payload.description_norm or payload.description or ""
    text_concat = (_norm_text(t) + " " + _norm_text(d)).strip()

    row = {
        "category": str(payload.category or "Unknown"),
        "condition": str(payload.condition or "Unknown"),
        "text_concat": text_concat,
    }
    return pd.DataFrame([row], columns=["category", "condition", "text_concat"])


def _choose_band(pred: float, category: str, condition: str) -> float:
    """
    Use per-(category, condition) q75 band if available; otherwise fallback to ±15% or min 100 PKR.
    Returns HALF-width of the interval to apply around the point estimate.
    """
    base = max(_INTERVAL_MIN, _INTERVAL_PCT * abs(float(pred)))

    if _calib_df is None:
        return base

    cat = str(category or "Unknown")
    cond = str(condition or "Unknown")
    hit = _calib_df[(_calib_df["category"] == cat) & (_calib_df["condition"] == cond)]
    if not hit.empty:
        q75 = float(hit["q75"].iloc[0])
        # Prevent too-narrow ranges on low-price items
        return max(q75, 0.10 * abs(float(pred)), 50.0)

    return base


def _blend_with_market(y_model: float, payload: PredictPriceIn) -> (float, str, int):
    """
    If market_stats are provided by backend, blend model prediction with median/avg
    and clamp to [min_price, max_price] when available.
    Returns (final_pred, blend_expl, confidence).
    """
    stats = payload.market_stats
    if not stats:
        # No stats: keep model pred, compat confidence ~68
        return y_model, "Compat-4 model (no market stats)", 68

    # Choose a baseline from stats
    baseline = None
    if stats.median_price is not None:
        baseline = float(stats.median_price)
    elif stats.avg_price is not None:
        baseline = float(stats.avg_price)

    # default: if both avg & median are missing, nothing to blend
    if baseline is None:
        return y_model, "Compat-4 model (market stats incomplete)", 68

    n = int(stats.sample_size or 0)

    # Weight: trust model more when sample is small; trust stats more when large
    # (<20: 0.65 model), (20-49: 0.55), (50-99: 0.45), (>=100: 0.35)
    if n < 20:
        w_model = 0.65
    elif n < 50:
        w_model = 0.55
    elif n < 100:
        w_model = 0.45
    else:
        w_model = 0.35

    y_blend = w_model * float(y_model) + (1.0 - w_model) * float(baseline)

    # Clamp within observed min/max if available
    lo_obs = float(stats.min_price) if stats.min_price is not None else None
    hi_obs = float(stats.max_price) if stats.max_price is not None else None
    if lo_obs is not None:
        y_blend = max(y_blend, lo_obs)
    if hi_obs is not None:
        y_blend = min(y_blend, hi_obs)

    # Confidence: base 68 (compat), +5 for n>=20, +5 more for n>=50, cap 90
    conf = 68
    if n >= 20:
        conf += 5
    if n >= 50:
        conf += 5
    conf = min(conf, 90)

    expl = f"Compat-4 × market blend (n={n}, w_model={w_model:.2f}, baseline={'median' if stats.median_price is not None else 'avg'})"
    return y_blend, expl, conf


def _interval_from_point(y: float, payload: PredictPriceIn) -> (float, float):
    """
    Construct min/max using calibrated band (q75) if available for the (category, condition),
    otherwise a simple ±max(MIN, PCT*y).
    """
    half = _choose_band(y, payload.category, payload.condition)
    lo = max(0.0, float(y) - half)
    hi = float(y) + half
    return (round(lo, 2), round(hi, 2))


# =========================
# Endpoint
# =========================
@router.post("/price-suggest", response_model=PredictPriceOut)
@timeboxed(settings.BUDGET_PRICE_MS)
def price_suggest(payload: PredictPriceIn):
    """
    Compat-4 price suggestion endpoint that matches backend's new ai.py:
    - Inputs: title/description + category + condition + optional market_stats
    - Model: LightGBM compat (category, condition, TF-IDF(title+desc))
    - Output: predicted_price + calibrated interval + confidence + explanation
    """
    m = load_model()
    X = _build_compat_row(payload)

    try:
        if hasattr(m, "predict"):
            y_model = float(m.predict(X)[0])
        elif isinstance(m, dict) and "pipeline" in m and hasattr(m["pipeline"], "predict"):
            y_model = float(m["pipeline"].predict(X)[0])
        else:
            raise TypeError("Unsupported model bundle for price prediction")

        # Blend with DB market stats if provided
        y_final, blend_expl, conf = _blend_with_market(y_model, payload)

        # Calibrated interval
        lo, hi = _interval_from_point(y_final, payload)

        expl = f"{blend_expl}; cat={X.loc[0,'category']} / cond={X.loc[0,'condition']}"
        return PredictPriceOut(
            predicted_price=round(y_final, 2),
            min_predicted_price=lo,
            max_predicted_price=hi,
            confidence=conf,
            explanation=expl,
        )

    except Exception:
        log.exception("price_suggest failed")
        # conservative fallback (zero-centered band)
        half = max(_INTERVAL_MIN, _INTERVAL_PCT * 0.0)
        return PredictPriceOut(
            predicted_price=0.0,
            min_predicted_price=0.0,
            max_predicted_price=round(half, 2),
            confidence=50,
            explanation="Fallback: model unavailable or input outside domain.",
        )
