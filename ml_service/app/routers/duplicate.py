from fastapi import APIRouter
from typing import List
from ..schemas import DuplicateIn, DuplicateOut
from ..config import settings
from ..utils import timeboxed
import joblib
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(tags=["Duplicate"])

_vec = None

def load_vec():
    global _vec
    if _vec is None:
        _vec = joblib.load(f"{settings.DUP_INDEX_DIR}/tfidf_vectorizer.joblib")
    return _vec

_ws_re = re.compile(r"\s+")

def norm_text(s: str) -> str:
    # very light normalization; keep it consistent with how you trained the vectorizer
    s = s.lower()
    s = s.replace("/", " ").replace("-", " ")
    s = _ws_re.sub(" ", s).strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", s).split() if t]

@router.post("/check-duplicate", response_model=DuplicateOut)
@timeboxed(settings.BUDGET_DUP_MS)
def check_duplicate(payload: DuplicateIn):
    vec = load_vec()

    cand_text_raw = f"{payload.title} {payload.description}"
    cand_text = norm_text(cand_text_raw)
    cand_tokens = tokens(cand_text)

    # Early exit if the candidate is too short/informationally poor
    if len(cand_tokens) < 3:
        return DuplicateOut(is_duplicate=False, confidence=0, similar_listing_ids=[])

    # Build corpus from existing listings, skipping exact text duplicates
    texts: List[str] = []
    keep_idx: List[int] = []
    for i, l in enumerate(payload.existing_listings):
        t = norm_text(f"{l.title} {l.description}")
        if t == cand_text:
            # skip tautology (candidate text exactly equals an existing text)
            continue
        texts.append(t)
        keep_idx.append(i)

    if not texts:
        return DuplicateOut(is_duplicate=False, confidence=0, similar_listing_ids=[])

    cand_vec = vec.transform([cand_text])
    mat = vec.transform(texts)
    sims = cosine_similarity(cand_vec, mat).ravel()  # [n_kept]

    # Top-1 / Top-2 stats
    order = np.argsort(-sims)
    top_sim = float(sims[order[0]])
    second_sim = float(sims[order[1]]) if sims.size > 1 else 0.0

    # Decision rule using configurable thresholds
    THI = settings.DUP_THRESH_HI
    TLO = settings.DUP_THRESH_LO
    MAR = settings.DUP_MARGIN

    is_dup = (top_sim >= THI) or (top_sim >= TLO and (top_sim - second_sim) >= MAR)

    # Only return ids above practical cutoff (≥ TLO)
    similar_ids: List[int | str] = []
    if is_dup:
        for idx in order:
            s = float(sims[idx])
            if s < TLO:
                break
            similar_ids.append(payload.existing_listings[keep_idx[idx]].id)
            if len(similar_ids) >= 10:
                break

    return DuplicateOut(
        is_duplicate=is_dup,
        confidence=int(round(top_sim * 100)),
        similar_listing_ids=similar_ids
    )
