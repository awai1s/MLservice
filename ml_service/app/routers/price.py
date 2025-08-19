from fastapi import APIRouter
from ..schemas import PredictPriceIn, PredictPriceOut
from ..config import settings
from ..utils import timeboxed
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

log = logging.getLogger("ml.price")
router = APIRouter(tags=["Price"])

# Heuristic interval settings (tunable later or move to config if you want)
_INTERVAL_MIN = 100.0     # PKR; minimum half-width
_INTERVAL_PCT = 0.15      # ±15% around point estimate

_model: Any = None
_seen_cats: Dict[str, set] = {}
_has_other: Dict[str, bool] = {}

# ---------- helpers to introspect training pipeline ----------
def _find_preprocessor(m: Any) -> Optional[ColumnTransformer]:
    pipe = None
    if isinstance(m, Pipeline):
        pipe = m
    elif isinstance(m, dict) and "pipeline" in m and isinstance(m["pipeline"], Pipeline):
        pipe = m["pipeline"]
    if pipe is None:
        return None

    for _, step in pipe.named_steps.items():
        if isinstance(step, ColumnTransformer):
            return step
        if isinstance(step, Pipeline):
            for __, sub in step.named_steps.items():
                if isinstance(sub, ColumnTransformer):
                    return sub
    return None

def _collect_seen_categories(pre: ColumnTransformer):
    global _seen_cats, _has_other
    _seen_cats.clear()
    _has_other.clear()
    try:
        for _, transformer, cols in pre.transformers_:
            enc = transformer
            if isinstance(transformer, Pipeline):
                for __, sub in transformer.named_steps.items():
                    if isinstance(sub, OneHotEncoder):
                        enc = sub
                        break
            if isinstance(enc, OneHotEncoder) and hasattr(enc, "categories_"):
                for col, cats in zip(cols, enc.categories_):
                    catset = set(cats.tolist() if hasattr(cats, "tolist") else list(cats))
                    _seen_cats[col] = catset
                    _has_other[col] = ("other" in catset) or ("Other" in catset)
    except Exception:
        log.exception("Failed to collect seen categories")

def _coerce_cat(col: str, val: str) -> str:
    if val is None:
        return "other" if _has_other.get(col, False) else next(iter(_seen_cats.get(col, {"unknown"})))
    sval = str(val).strip()
    cats = _seen_cats.get(col)
    if not cats:
        return sval
    if sval in cats:
        return sval
    if col == "brand":
        low = sval.lower()
        brand_map = {
            "iphone": "Apple", "apple": "Apple",
            "samsung": "Samsung", "xiaomi": "Xiaomi",
            "dell": "Dell", "hp": "HP", "h p": "HP", "lenovo": "Lenovo",
            "asus": "ASUS", "acer": "Acer", "huawei": "Huawei",
        }
        for k, v in brand_map.items():
            if k in low and v in cats:
                return v
    return "other" if _has_other.get(col, False) else next(iter(cats))

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.PRICE_MODEL_PATH)
        log.info("Loaded price model: %s", type(_model))
        pre = _find_preprocessor(_model if hasattr(_model, "predict") else _model.get("pipeline"))
        if pre is not None:
            _collect_seen_categories(pre)
            log.info("Seen categorical features: %s", list(_seen_cats.keys()))
        else:
            log.warning("No ColumnTransformer found; category coercion disabled.")
    return _model

# ---------- request -> DataFrame ----------
def _build_row(payload: PredictPriceIn) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "title": payload.title,
        "description": payload.description,
        "category": payload.category,
        "condition": payload.condition,
        "status": "active",
        "meeting_option": "on-campus",
        "shipping_option": "meetup",
        "category_path": payload.category,
        "brand": payload.title,            # brand inferred from title if possible
        "university": "unknown",
        "university_email_domain": "unknown",
        "negotiable": True,
        "has_image": True,
        "has_brand": False,
        # let imputers handle NaN rather than hard zeros
        "listing_age_days": np.nan,
        "seller_rating": np.nan,
        "seller_num_sales": np.nan,
        "views": np.nan,
        "likes": np.nan,
        "saved_count": np.nan,
        "discount_pct_clean": np.nan,
        "price_original": np.nan,
    }

    for col in [
        "category", "condition", "status", "meeting_option",
        "shipping_option", "category_path", "brand",
        "university", "university_email_domain",
    ]:
        row[col] = _coerce_cat(col, row.get(col))

    row["has_brand"] = bool(row.get("brand") and row["brand"] not in ("other", "Other", "unknown"))

    df = pd.DataFrame([row])
    for c in ["negotiable", "has_image", "has_brand"]:
        df[c] = df[c].astype(bool)
    return df

def _interval_from_point(y: float) -> tuple[float, float]:
    """Return (min, max) around y using a simple ±max(MIN, PCT*y) band."""
    half = max(_INTERVAL_MIN, _INTERVAL_PCT * abs(y))
    lo = max(0.0, y - half)
    hi = y + half
    return (round(lo, 2), round(hi, 2))

# ---------- endpoint ----------
@router.post("/predict-price", response_model=PredictPriceOut)
@timeboxed(settings.BUDGET_PRICE_MS)
def predict_price(payload: PredictPriceIn):
    m = load_model()
    X = _build_row(payload)

    try:
        if hasattr(m, "predict"):
            y = float(m.predict(X)[0])
        elif isinstance(m, dict) and "pipeline" in m and hasattr(m["pipeline"], "predict"):
            y = float(m["pipeline"].predict(X)[0])
        elif isinstance(m, dict) and "preprocessor" in m and "estimator" in m:
            Xp = m["preprocessor"].transform(X)
            y = float(m["estimator"].predict(Xp)[0])
        else:
            raise TypeError("Unsupported model bundle for price prediction")

        # build simple interval
        lo, hi = _interval_from_point(y)
        conf = 85
        expl = f"Based on {X.loc[0,'category']} / {X.loc[0,'condition']} with brand={X.loc[0,'brand']}"

        return PredictPriceOut(
            predicted_price=round(y, 2),
            min_predicted_price=lo,
            max_predicted_price=hi,
            confidence=conf,
            explanation=expl,
        )
    except Exception:
        log.exception("price predict failed")
        # conservative fallback (zero-centered band)
        lo, hi = _interval_from_point(0.0)
        return PredictPriceOut(
            predicted_price=0.0,
            min_predicted_price=lo,
            max_predicted_price=hi,
            confidence=50,
            explanation="Fallback: model unavailable or input outside training domain.",
        )
