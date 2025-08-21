from fastapi import APIRouter
from typing import List, Optional
from ..schemas import RecommendIn, RecommendOut, Recommendation
from ..config import settings
from ..utils import timeboxed

import os, json, joblib, re
import numpy as np

router = APIRouter(tags=["Recommend"])

# ---------- text utils ----------
_ws = re.compile(r"\s+")
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    return _ws.sub(" ", s)

def _token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", _norm(s)))

def _title_sim(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

def _scale01_to_pct(x: float) -> int:
    x = max(0.0, min(1.0, x))
    return int(round(100 * x))

# ---------- ALS lazy-load ----------
_als_model = None
_als_maps: Optional[dict] = None

def _load_als():
    """Load ALS model + mappings only if both files exist; else return (None, None)."""
    global _als_model, _als_maps
    if _als_model is not None or _als_maps is not None:
        return _als_model, _als_maps

    if not os.path.exists(settings.RECO_MODEL_PATH):
        return None, None
    if not os.path.exists(settings.RECO_MAPPINGS):
        return None, None

    try:
        _als_model = joblib.load(settings.RECO_MODEL_PATH)  # implicit ALS or compatible
    except Exception:
        _als_model = None

    try:
        with open(settings.RECO_MAPPINGS, "r", encoding="utf-8") as f:
            _als_maps = json.load(f)  # expects keys: "user2idx", "item2idx"
    except Exception:
        _als_maps = None

    return _als_model, _als_maps

@router.post("/recommend", response_model=RecommendOut)
@timeboxed(settings.BUDGET_RECO_MS)
def recommend(payload: RecommendIn):
    """
    Hybrid recommender:
      - Baseline: content (title) + filters (category/condition) + popularity (likes/saves/views)
      - ALS personalization: only if model + mappings + user_id are available; blended with baseline.
    """
    cand_cat  = (payload.category or "").strip()
    cand_cond = (payload.condition or "").strip()
    cand_title = payload.title or ""
    cand_desc  = payload.description or ""
    cand_text  = f"{cand_title} {cand_desc}".strip()

    listings = payload.available_listings or []
    if not listings:
        return RecommendOut(recommendations=[], reasoning="No available listings provided.")

    # ---------- Popularity for the current pool (normalized 0..1) ----------
    pop_raw = []
    for l in listings:
        s = float(getattr(l, "saved_count", 0) or 0)
        k = float(getattr(l, "likes", 0) or 0)
        v = float(getattr(l, "views", 0) or 0)
        pop_raw.append(0.4 * s + 0.4 * k + 0.2 * v)

    pmin, pmax = (min(pop_raw), max(pop_raw)) if pop_raw else (0.0, 1.0)
    pop_norm = [(x - pmin) / (pmax - pmin) if pmax > pmin else 0.5 for x in pop_raw]

    # ---------- Baseline score (0..1) ----------
    # weights: title 0.40, category 0.25, condition 0.15, popularity 0.20
    base_scores = []
    cat_matches = []
    cond_matches = []
    title_sims = []
    for idx, l in enumerate(listings):
        cat_match  = 1.0 if (l.category or "") == cand_cat and cand_cat else 0.0
        cond_match = 1.0 if (l.condition or "") == cand_cond and cand_cond else 0.0
        tsim       = _title_sim(cand_text, f"{l.title or ''} {l.description or ''}")
        pop01      = float(pop_norm[idx])

        base = (0.40 * tsim) + (0.25 * cat_match) + (0.15 * cond_match) + (0.20 * pop01)
        base = max(0.0, min(1.0, base))

        base_scores.append(base)
        cat_matches.append(cat_match)
        cond_matches.append(cond_match)
        title_sims.append(tsim)

    base_scores = np.array(base_scores, dtype=float)

    # ---------- ALS personalization (only if everything is available) ----------
    als_scores = None
    model, maps = _load_als()
    if (
        model is not None
        and maps is not None
        and getattr(payload, "user_id", None) is not None
        and "user2idx" in maps
        and "item2idx" in maps
    ):
        uid_str = str(payload.user_id)
        if uid_str in maps["user2idx"]:
            try:
                uidx = int(maps["user2idx"][uid_str])
                user_vec = model.user_factors[uidx]  # shape [k]

                # score only the provided pool (requires each listing id mapped to ALS index)
                raw = []
                for l in listings:
                    iid = str(l.id)
                    if iid in maps["item2idx"]:
                        iidx = int(maps["item2idx"][iid])
                        item_vec = model.item_factors[iidx]  # shape [k]
                        raw.append(float(np.dot(user_vec, item_vec)))  # raw CF score
                    else:
                        raw.append(None)

                # normalize ALS scores over the available candidates
                vals = [x for x in raw if x is not None]
                if vals:
                    amin, amax = min(vals), max(vals)
                    als_scores = []
                    for x in raw:
                        if x is None:
                            als_scores.append(0.0)
                        else:
                            als_scores.append((x - amin) / (amax - amin + 1e-9) if amax > amin else 0.5)
                    als_scores = np.array(als_scores, dtype=float)
                # else: keep als_scores=None to skip blending
            except Exception:
                als_scores = None  # fail-safe

    # ---------- Blend (if ALS available) ----------
    ALS_WEIGHT = 0.35  # tune as needed
    if als_scores is not None:
        final01 = (1.0 - ALS_WEIGHT) * base_scores + ALS_WEIGHT * als_scores
        personalized_flags = als_scores > 0.0
    else:
        final01 = base_scores
        personalized_flags = np.zeros_like(final01, dtype=bool)

    # ---------- Build response ----------
    scored = []
    for idx, l in enumerate(listings):
        reasons = []
        if cat_matches[idx] > 0: reasons.append(f"same category {l.category}")
        if cond_matches[idx] > 0: reasons.append(f"same condition {l.condition}")
        if title_sims[idx] >= 0.40: reasons.append("title looks similar")
        if pop_norm[idx] >= 0.66: reasons.append("popular on campus")
        if personalized_flags[idx]: reasons.append("personalized for you")

        pct = int(round(100 * max(0.0, min(1.0, float(final01[idx])))))
        scored.append((pct, Recommendation(listing_id=l.id, score=pct, reason=", ".join(reasons) if reasons else "relevant match")))

    scored.sort(key=lambda x: -x[0])
    top = [rec for _, rec in scored[:10]]

    why = []
    if cand_cat:  why.append(f"matching category “{cand_cat}”")
    if cand_cond: why.append(f"matching condition “{cand_cond}”")
    if cand_title or cand_desc: why.append("similar titles/descriptions")
    if any("popular on campus" in r.reason for r in top):
        why.append("boosted by item popularity")
    if any("personalized for you" in r.reason for r in top):
        why.append("personalized using your past activity")
    if not why:  why.append("overall relevance")

    return RecommendOut(recommendations=top, reasoning="; ".join(why) + ".")
