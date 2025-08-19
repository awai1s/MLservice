from pydantic import BaseModel, Field
from typing import List, Union

# ---------- /predict-price ----------
class PredictPriceIn(BaseModel):
    title: str
    description: str
    category: str
    condition: str

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


# ---------- /recommend ----------
from pydantic import BaseModel, Field
from typing import List, Union

# ---------- /recommend ----------
class UserPreferences(BaseModel):
    categories: List[str]
    max_price: float  # required

class RecommendIn(BaseModel):
    user_preferences: UserPreferences  # <-- only user preferences

class Recommendation(BaseModel):
    listing_id: Union[int, str]
    score: int = Field(ge=0, le=100)
    reason: str

class RecommendOut(BaseModel):
    recommendations: List[Recommendation]
    reasoning: str  # top-level summary for the response

