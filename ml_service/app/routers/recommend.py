from fastapi import APIRouter
from typing import List
from ..schemas import RecommendIn, RecommendOut, Recommendation
from ..config import settings
from ..utils import timeboxed
import joblib, json, math

router = APIRouter(tags=["Recommend"])

_model = None
_maps = None

def _load():
    global _model, _maps
    if _model is None:
        _model = joblib.load(settings.RECO_MODEL_PATH)   # ALS model
        with open(settings.RECO_MAPPINGS, "r", encoding="utf-8") as f:
            _maps = json.load(f)  # expects user_to_idx, item_to_idx, idx_to_item
    return _model, _maps

def _scale01_to_pct(x: float) -> int:
    x = max(0.0, min(1.0, x))
    return int(round(100 * x))

@router.post("/recommend", response_model=RecommendOut)
@timeboxed(settings.BUDGET_RECO_MS)
def recommend(payload: RecommendIn):
    """
    Rank ONLY the provided available_listings based on:
    - ALS user->item signal (if user known)
    - Category match to user_preferences.categories
    - Price fit vs. user_preferences.max_price
    Returns top items with score 0..100 and a brief reason.
    """
    model, maps = _load()
    uid = maps.get("user_to_idx", {}).get(str(payload.user_id))
    have_user = uid is not None

    recs: List[Recommendation] = []
    cats = set(payload.user_preferences.categories)
    max_price = payload.user_preferences.max_price  # required by schema (can still be 0)

    for lst in payload.available_listings:
        reason_bits = []
        score_components = []

        # (1) ALS signal
        als_score = 0.0
        if have_user:
            item_idx = maps.get("item_to_idx", {}).get(str(lst.id))
            if item_idx is not None:
                u = model.user_factors[uid]
                v = model.item_factors[item_idx]
                dot = float((u * v).sum())
                als_score = 1 / (1 + math.exp(-dot))  # squash to 0..1
                score_components.append(0.6 * als_score)
                reason_bits.append("similar to items you interacted with")

        # (2) Category preference
        cat_bonus = 1.0 if (lst.category in cats) else 0.0
        score_components.append(0.25 * cat_bonus)
        if cat_bonus > 0:
            reason_bits.append(f"matches preferred category {lst.category}")

        # (3) Price fit (guard against zero/negative to avoid division by zero)
        price_bonus = 0.0
        if max_price > 0:
            if lst.price <= max_price:
                rel = 1.0 - (max_price - lst.price) / max_price  # 0..1
                rel = max(0.0, min(1.0, rel))
                price_bonus = 0.15 * rel
                reason_bits.append(f"within your budget (≤ {max_price})")
            else:
                price_bonus = -0.1
                reason_bits.append("over budget")
        # else: no budget contribution

        score_components.append(price_bonus)

        score01 = max(0.0, min(1.0, sum(score_components)))
        recs.append(Recommendation(
            listing_id=lst.id,
            score=_scale01_to_pct(score01),
            reason=", ".join(reason_bits) if reason_bits else "popular on campus",
        ))

    recs.sort(key=lambda r: r.score, reverse=True)
    return RecommendOut(recommendations=recs[:10])
