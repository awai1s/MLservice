from fastapi import APIRouter
from ..schemas import PredictPriceIn, PredictPriceOut
from ..config import settings
from ..utils import timeboxed
import joblib
import logging
import pandas as pd
from typing import Any, Dict

log = logging.getLogger("ml.price")

router = APIRouter(tags=["Price"])

_model: Any = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(settings.PRICE_MODEL_PATH)
        log.info("Loaded price model: %s", type(_model))
        if isinstance(_model, dict):
            log.info("Model bundle keys: %s", list(_model.keys()))
    return _model

def _build_row(payload: PredictPriceIn) -> pd.DataFrame:
    """
    Build a single-row DataFrame that includes EVERY column your ColumnTransformer expects.
    Backend only sends: title, description, category, condition
    We fill the rest with safe defaults so the pipeline won't crash.
    """
    # Core from schema (required)
    row: Dict[str, Any] = {
        "title": payload.title,
        "description": payload.description,
        "category": payload.category,
        "condition": payload.condition,
    }

    # Defaults for missing training-time features (from the error list + your Day 3 notes)
    # Strings
    row.update({
        "status": "active",
        "meeting_option": "on-campus",
        "shipping_option": "meet",
        "category_path": payload.category,          # simple proxy
        "brand": "",                                # no brand in schema
        "university": "",                           # no university in schema
        "university_email_domain": "",              # no email domain in schema
    })
    # Booleans
    row.update({
        "negotiable": False,
        "has_image": False,
        "has_brand": False,                         # derived from brand presence
    })
    # Numerics (ints/floats)
    row.update({
        "listing_age_days": 0,
        "seller_rating": 4.5,
        "seller_num_sales": 0,
        "views": 0,
        "likes": 0,
        "saved_count": 0,
        "discount_pct_clean": 0.0,
        "price_original": 0.0,                      # avoid None → imputers often expect numeric
    })

    # Derive has_brand from brand text, just in case you start passing it later:
    row["has_brand"] = bool(row["brand"])

    # Ensure DataFrame with correct dtypes (helps ColumnTransformer)
    df = pd.DataFrame([row])
    # Cast common numeric types explicitly (optional but safer)
    num_cols = [
        "listing_age_days", "seller_rating", "seller_num_sales", "views", "likes",
        "saved_count", "discount_pct_clean", "price_original"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    bool_cols = ["negotiable", "has_image", "has_brand"]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    return df

@router.post("/predict-price", response_model=PredictPriceOut)
@timeboxed(settings.BUDGET_PRICE_MS)
def predict_price(payload: PredictPriceIn):
    """
    Expects: {title, description, category, condition}
    Returns: {predicted_price, confidence(0..100), explanation}
    """
    m = load_model()
    X = _build_row(payload)

    try:
        # CASE A: full sklearn Pipeline
        if hasattr(m, "predict"):
            y = float(m.predict(X)[0])

        # CASE B: dict bundle with a 'pipeline'
        elif isinstance(m, dict) and "pipeline" in m and hasattr(m["pipeline"], "predict"):
            y = float(m["pipeline"].predict(X)[0])

        # CASE C: dict bundle with 'preprocessor' + 'estimator'
        elif isinstance(m, dict) and "preprocessor" in m and "estimator" in m:
            Xp = m["preprocessor"].transform(X)
            y = float(m["estimator"].predict(Xp)[0])

        else:
            raise TypeError(f"Unsupported model bundle: {type(m)} keys={list(m.keys()) if isinstance(m, dict) else 'N/A'}")

        conf = 85  # heuristic; replace with calibrated confidence if available
        expl = f"Based on {payload.category} in {payload.condition} condition and similar listings."
        return PredictPriceOut(predicted_price=round(y, 2), confidence=conf, explanation=expl)

    except Exception:
        log.exception("price predict failed")
        return PredictPriceOut(
            predicted_price=0.0,
            confidence=50,
            explanation="Fallback: model unavailable or input outside training domain.",
        )
