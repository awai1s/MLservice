from fastapi import APIRouter
from typing import List
from ..schemas import DuplicateIn, DuplicateOut
from ..config import settings
from ..utils import timeboxed

import os
import joblib
import numpy as np
import pandas as pd
import re
from difflib import SequenceMatcher
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(tags=["Duplicate"])

# -------------------------
# Globals (lazy-loaded)
# -------------------------
_vec = None
_mat = None
_meta = None

def load_index():
    """Load TF-IDF vectorizer, matrix, and item metadata once (lazy singleton)."""
    global _vec, _mat, _meta
    if _vec is None or _mat is None or _meta is None:
        vec_path = os.path.join(settings.DUP_INDEX_DIR, "tfidf_vectorizer.joblib")
        mat_path = os.path.join(settings.DUP_INDEX_DIR, "tfidf_matrix.npz")
        meta_path = os.path.join(settings.DUP_INDEX_DIR, "item_meta.csv")

        _vec = joblib.load(vec_path)
        _mat = sparse.load_npz(mat_path)
        _meta = pd.read_csv(meta_path)

    return _vec, _mat, _meta

# -------------------------
# Text utils
# -------------------------
_ws_re = re.compile(r"\s+")
def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("/", " ").replace("-", " ")
    s = _ws_re.sub(" ", s).strip()
    return s

# -------------------------
# Core route
# -------------------------
@router.post("/check-duplicate", response_model=DuplicateOut)
@timeboxed(settings.BUDGET_DUP_MS)
def check_duplicate(payload: DuplicateIn):
    """
    Hybrid duplicate checker:
    - If payload.existing_listings provided: use SequenceMatcher LIKE backend example.
    - Else: compare against prebuilt TF-IDF index for entire catalog.
    Returns IDs of similar listings and duplicate flag using thresholds.
    """
    THI = settings.DUP_THRESH_HI
    TLO = settings.DUP_THRESH_LO
    MAR = settings.DUP_MARGIN

    cand_title = norm_text(payload.title)
    cand_desc  = norm_text(payload.description)
    cand_text  = f"{cand_title} {cand_desc}".strip()

    # Early exit if too short
    if len(cand_text.split()) < 3:
        return DuplicateOut(is_duplicate=False, confidence=0, similar_listing_ids=[])

    # --------------- Mode A: use provided existing_listings (SequenceMatcher) ---------------
    if getattr(payload, "existing_listings", None):
        scores = []
        for l in payload.existing_listings:
            lt = norm_text(l.title or "")
            # Basic title-only similarity (matches your example); optionally mix description
            sim = SequenceMatcher(None, cand_title, lt).ratio()
            scores.append((l.id, sim))

        if not scores:
            return DuplicateOut(is_duplicate=False, confidence=0, similar_listing_ids=[])

        # Sort by similarity desc
        scores.sort(key=lambda x: -x[1])
        top_id, top_sim = scores[0]
        second_sim = scores[1][1] if len(scores) > 1 else 0.0

        is_dup = (top_sim >= THI) or (top_sim >= TLO and (top_sim - second_sim) >= MAR)

        # Return up to 10 IDs with sim >= TLO
        similar_ids: List[int | str] = [sid for sid, s in scores if s >= TLO][:10]

        return DuplicateOut(
            is_duplicate=is_dup,
            confidence=int(round(top_sim * 100)),
            similar_listing_ids=similar_ids
        )

    # --------------- Mode B: fallback to global TF-IDF index ---------------
    vec, mat, meta = load_index()
    cand_vec = vec.transform([cand_text])
    sims = cosine_similarity(cand_vec, mat).ravel()

    order = np.argsort(-sims)
    top_sim = float(sims[order[0]])
    second_sim = float(sims[order[1]]) if sims.size > 1 else 0.0

    is_dup = (top_sim >= THI) or (top_sim >= TLO and (top_sim - second_sim) >= MAR)

    similar_ids: List[int | str] = []
    for idx in order:
        s = float(sims[idx])
        if s < TLO:
            break
        similar_ids.append(meta.iloc[idx]["id"])  # assumes 'id' column in item_meta.csv
        if len(similar_ids) >= 10:
            break

    return DuplicateOut(
        is_duplicate=is_dup,
        confidence=int(round(top_sim * 100)),
        similar_listing_ids=similar_ids
    )
