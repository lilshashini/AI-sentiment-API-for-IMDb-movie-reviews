from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    text: str = Field(..., example="I absolutely love how fast the delivery was!")

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

# --- NEW SCHEMA FOR THE BONUS ---
class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., example=["I loved it!", "It was terrible.", "It was just okay."])