# app/config.py
import os
from dataclasses import dataclass

import redis

@dataclass
class Settings:
    # app
    APP_NAME: str = os.getenv("APP_NAME", "Campus Exchange ML Service")
    PORT: int = int(os.getenv("PORT", "9000"))

    # latency budgets (ms)
    BUDGET_PRICE_MS: int = int(os.getenv("BUDGET_PRICE_MS", "300"))
    BUDGET_DUP_MS: int = int(os.getenv("BUDGET_DUP_MS", "500"))
    BUDGET_RECO_MS: int = int(os.getenv("BUDGET_RECO_MS", "300"))

    # duplicate detection thresholds (tunable without code deploy)
    DUP_THRESH_HI: float = float(os.getenv("DUP_THRESH_HI", "0.88"))
    DUP_THRESH_LO: float = float(os.getenv("DUP_THRESH_LO", "0.80"))
    DUP_MARGIN: float    = float(os.getenv("DUP_MARGIN",    "0.06"))

    # model paths
    PRICE_MODEL_PATH: str = os.getenv(
        "PRICE_MODEL_PATH",
        "app/models/price_model_compat4_v2.joblib"
    )
    CALIB_PATH: str = os.getenv(
        "CALIB_PATH",
        "app/models/error_bands_by_cat_cond.csv"
    )

    # duplicate index (vectorizer + tfidf matrix + item_meta)
    DUP_INDEX_DIR: str = os.getenv(
        "DUP_INDEX_DIR",
        "app/models/dupdet_index"
    )

    # recommendation models
    RECO_MODEL_PATH: str = os.getenv(
        "RECO_MODEL_PATH",
        "app/models/reco/als_bm25_tuned.joblib"
    )
    RECO_ITEM_META: str = os.getenv(
        "RECO_ITEM_META",
        "app/models/reco/item_catalog.csv"
    )
    RECO_MAPPINGS: str = os.getenv(
        "RECO_MAPPINGS",
        "app/models/reco/mappings.json"
    )

    # caching (optional). empty REDIS_URL means "disabled"
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/2")
    CACHE_TTL_S: int = int(os.getenv("CACHE_TTL_S", "600"))

    # logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
