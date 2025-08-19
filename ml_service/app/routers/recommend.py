from fastapi import APIRouter
from typing import List, Optional
from ..schemas import RecommendIn, RecommendOut, Recommendation
from ..config import settings
from ..utils import timeboxed
import joblib, json, math, os
import pandas as pd
import numpy as np

router = APIRouter(tags=["Recommend"])

# Optional: keep handles if you later want ALS/item factors
_model = None
_maps = None
_catalog: Optional[pd.DataFrame] = None

def _load_models():
    """
    Optional ALS + mappings loader (kept for future use).
    We don't require them for the 'preferences-only' contract,
    but we try to load gracefully if present.
    """
    global _model, _maps
    if _model is None and os.path.exists(settings.RECO_MODEL_PATH):
        try:
            _model = joblib.load(settings.RECO_MODEL_PATH)
        except Exception:
            _model = None
    if _maps is None and os.path.exists(settings.RECO_MAPPINGS):
        try:
            with open(settings.RECO_MAPPINGS, "r", encoding="utf-8") as f:
                _maps = json.load(f)
        except Exception:
            _maps = None
    return _model, _maps

def _load_catalog() -> pd.DataFrame:
    """
    Load item catalog with columns like:
      - id or item_id
      - title (optional)
      - category
      - price
      - pop (optional popularity score)
    Falls back sensibly if some columns are missing.
    """
    global _catalog
    if _catalog is not None:
        return _catalog

    if not os.path.exists(settings.RECO_ITEM_META):
        # empty fallback
        _catalog = pd.DataFrame(columns=["listing_id", "category", "price", "pop"])
        return _catalog

    df = pd.read_csv(settings.RECO_ITEM_META)

    # Normalize id column name
    if "listing_id" not in df.columns:
        if "item_id" in df.columns:
            df = df.rename(columns={"item_id": "listing_id"})
        elif "id" in df.columns:
            df = df.rename(columns={"id": "listing_id"})
        else:
            # as a last resort, use row index as id
            df["listing_id"] = np.arange(len(df))

    # Ensure required fields exist
    if "category" not in df.columns:
        df["category"] = "Other"
    if "price" not in df.columns:
        df["price"] = np.nan
    if "pop" not in df.columns:
        # if we have any engagement columns, build a crude popularity; else 1.0
        pop_cols = [c for c in ["saved_count", "likes", "views"] if c in df.columns]
        if pop_cols:
            df["pop"] = df[pop_cols].fillna(0).sum(axis=1)
        else:
            df["pop"] = 1.0

    # Clean price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median() if df["price"].notna().any() else 0.0)

    _catalog = df
    return _catalog

def _scale01_to_pct(x: float) -> int:
    x = max(0.0, min(1.0, x))
    return int(round(100 * x))

@router.post("/recommend", response_model=RecommendOut)
@timeboxed(settings.BUDGET_RECO_MS)
def recommend(payload: RecommendIn):
    """
    Input: only user_preferences {categories, max_price}
    Output: top-N recommendations + a top-level reasoning string.
    """
    _load_models()  # not strictly needed, but harmless
    df = _load_catalog().copy()

    cats_pref = set([c.strip() for c in payload.user_preferences.categories if c.strip()])
    max_price = float(payload.user_preferences.max_price)

    # ------- Filter candidates -------
    filtered = df
    filters_applied = []

    if cats_pref:
        filtered = filtered[filtered["category"].isin(cats_pref)]
        filters_applied.append(f"categories={sorted(cats_pref)}")

    if max_price > 0:
        filtered = filtered[filtered["price"] <= max_price]
        filters_applied.append(f"max_price≤{max_price}")

    # If filters removed everything, relax them gradually
    if filtered.empty and cats_pref:
        filtered = df[df["category"].isin(cats_pref)]
    if filtered.empty:
        filtered = df  # fallback to whole catalog

    if filtered.empty:
        return RecommendOut(recommendations=[], reasoning="No items available in catalog.")

    # ------- Scoring -------
    # Popularity component
    pop = filtered["pop"].astype(float)
    pmin, pmax = float(pop.min()), float(pop.max())
    pop_norm = (pop - pmin) / (pmax - pmin) if pmax > pmin else pd.Series(0.5, index=filtered.index)

    # Budget fit component
    if max_price > 0:
        fit = 1.0 - (max_price - filtered["price"]) / max_price
        fit = fit.clip(lower=0.0, upper=1.0)
    else:
        fit = pd.Series(0.0, index=filtered.index)

    # Category bonus (small boost if matches preference)
    cat_bonus = filtered["category"].isin(cats_pref).astype(float) * 0.05

    # Weighted sum → 0..1
    score01 = (0.75 * pop_norm) + (0.20 * fit) + cat_bonus
    score01 = score01.clip(lower=0.0, upper=1.0)

    filtered = filtered.assign(_score=score01)

    # ------- Build response -------
    filtered = filtered.sort_values("_score", ascending=False).head(10)

    recs: List[Recommendation] = []
    for _, row in filtered.iterrows():
        reasons = []
        if row["_score"] >= 0:
            if cats_pref and row["category"] in cats_pref:
                reasons.append(f"matches preferred category {row['category']}")
            if max_price > 0:
                if row["price"] <= max_price:
                    reasons.append(f"within your budget (≤ {max_price})")
                else:
                    reasons.append("slightly above budget")
            # popularity reasoning
            if pmax > pmin and row["pop"] >= (pmin + 0.66 * (pmax - pmin)):
                reasons.append("popular on campus")

        recs.append(
            Recommendation(
                listing_id=row["listing_id"],
                score=_scale01_to_pct(float(row["_score"])),
                reason=", ".join(reasons) if reasons else "recommended by popularity"
            )
        )

    summary_bits = []
    if cats_pref:
        summary_bits.append(f"focused on categories {sorted(cats_pref)}")
    if max_price > 0:
        summary_bits.append(f"priced at or under {max_price}")
    if not summary_bits:
        summary_bits.append("ranked by popularity")
    reasoning = "; ".join(summary_bits) + "."

    return RecommendOut(recommendations=recs, reasoning=reasoning)
