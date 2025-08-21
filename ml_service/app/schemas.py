from pydantic import BaseModel, Field
from typing import List, Union

from pydantic import BaseModel, ConfigDict
from typing import Optional


# ---------- Shared ----------
class BaseLooseModel(BaseModel):
    # ignore any unexpected keys; prevents 422 even if backend adds fields
    model_config = ConfigDict(extra='ignore')


# ---------- /price-suggest ----------
class MarketStats(BaseLooseModel):
    avg_price: Optional[float] = None
    median_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    sample_size: Optional[int] = 0


class PredictPriceIn(BaseLooseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    # If frontend sends normalized text, we'll prefer it; otherwise we normalize here.
    title_norm: Optional[str] = None
    description_norm: Optional[str] = None

    category: str
    condition: str

    # New: backend now passes DB-driven stats for the category
    market_stats: Optional[MarketStats] = None


class PredictPriceOut(BaseModel):
    predicted_price: float          # e.g., 450.0
    min_predicted_price: float      # lower bound of price band
    max_predicted_price: float      # upper bound of price band
    confidence: int                 # 0..100
    explanation: str


# ---------- /check-duplicate ----------
class ExistingListing(BaseModel):
    id: Union[int, str]
    title: str
    description: str

class DuplicateIn(BaseModel):
    title: str
    description: str
    existing_listings: List[ExistingListing]

class DuplicateOut(BaseModel):
    is_duplicate: bool
    confidence: int = Field(ge=0, le=100)
    similar_listing_ids: List[Union[int, str]]


from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

# Let inputs accept extra keys safely
class BaseLooseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

# ---------- /recommend ----------
class RecommendListingItem(BaseLooseModel):
    id: Union[str, int]
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    condition: Optional[str] = None
    # OPTIONAL engagement for popularity scoring
    likes: Optional[int] = None
    saved_count: Optional[int] = None
    views: Optional[int] = None

class RecommendIn(BaseLooseModel):
    # OPTIONAL user_id enables ALS personalization when mappings+model exist
    user_id: Optional[Union[str, int]] = None

    # Candidate context from backend
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    condition: Optional[str] = None

    # Pool to rank
    available_listings: List[RecommendListingItem]

class Recommendation(BaseModel):
    listing_id: Union[int, str]
    score: int = Field(ge=0, le=100)
    reason: str

class RecommendOut(BaseModel):
    recommendations: List[Recommendation]
    reasoning: str


