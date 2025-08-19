from pydantic import BaseModel, Field
from typing import List, Union

# ---------- /predict-price ----------
class PredictPriceIn(BaseModel):
    title: str
    description: str
    category: str
    condition: str

class PredictPriceOut(BaseModel):
    predicted_price: float
    confidence: int = Field(ge=0, le=100)
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
class UserPreferences(BaseModel):
    categories: List[str]
    max_price: float  # require it (no Optional)

class AvailableListing(BaseModel):
    id: Union[int, str]
    title: str
    category: str
    price: float

class RecommendIn(BaseModel):
    user_id: Union[int, str]
    user_preferences: UserPreferences
    available_listings: List[AvailableListing]

class Recommendation(BaseModel):
    listing_id: Union[int, str]
    score: int = Field(ge=0, le=100)
    reason: str

class RecommendOut(BaseModel):
    recommendations: List[Recommendation]
