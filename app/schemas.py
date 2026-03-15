"""
Pydantic request and response models for the sentiment analysis API.

These models provide:
- Request body validation and documentation
- Response serialization with type safety
- Automatic OpenAPI/Swagger documentation
"""

from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.
    
    Validates and documents the input for the /predict endpoint.
    
    Attributes:
        text (str): The review or text to analyze for sentiment.
                   Must be a non-empty string.
    
    Example:
        {
            "text": "I absolutely love how fast the delivery was!"
        }
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="I absolutely love how fast the delivery was!",
        description="The text to analyze for sentiment (1-5000 characters)"
    )


class PredictionResponse(BaseModel):
    """
    Response model for sentiment prediction.
    
    Contains the prediction result with sentiment classification and confidence.
    
    Attributes:
        text (str): The original input text that was analyzed.
        sentiment (str): The predicted sentiment class.
                        One of: "positive", "negative", "neutral"
        confidence (float): Confidence score from 0.0 to 1.0,
                          rounded to 2 decimal places.
    
    Example:
        {
            "text": "I absolutely love how fast the delivery was!",
            "sentiment": "positive",
            "confidence": 0.94
        }
    """
    text: str = Field(
        ...,
        description="The original input text"
    )
    sentiment: str = Field(
        ...,
        description="Predicted sentiment: 'positive', 'negative', or 'neutral'"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch sentiment predictions.
    
    Allows processing multiple texts in a single request for efficiency.
    
    Attributes:
        texts (List[str]): List of review texts to analyze.
                          Each text must be 1-5000 characters.
                          Maximum 100 texts per request.
    
    Example:
        {
            "texts": [
                "I love this product!",
                "This is terrible.",
                "It's okay, nothing special."
            ]
        }
    """
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to analyze (1-100 texts)"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch sentiment predictions.
    
    Contains predictions for all submitted texts.
    
    Attributes:
        results (List[PredictionResponse]): List of prediction results,
                                           one per input text in the same order.
        count (int): Total number of predictions in the results.
    
    Example:
        {
            "results": [
                {
                    "text": "I love this product!",
                    "sentiment": "positive",
                    "confidence": 0.95
                },
                {
                    "text": "This is terrible.",
                    "sentiment": "negative",
                    "confidence": 0.92
                }
            ],
            "count": 2
        }
    """
    results: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )
    count: int = Field(
        ...,
        ge=1,
        description="Total number of predictions"
    )
