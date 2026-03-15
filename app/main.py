"""
FastAPI application for sentiment analysis.

This module provides:
- Health check endpoint to verify the service is running
- Prediction endpoint to classify text sentiment
- Automatic model loading on application startup
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app import model as sentiment_model
import sklearn  # Required for unpickling scikit-learn models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles:
    - Loading the sentiment model when the server starts
    - Cleaning up resources when the server shuts down
    """
    # Load the pre-trained sentiment model from disk
    try:
        sentiment_model.load_model("model/sentiment_model.pkl")
        print("✅ Sentiment model loaded successfully on startup!")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        raise
    
    # Server is running - yield control back to FastAPI
    yield
    
    # Cleanup: Reset the model when the server shuts down
    print("🛑 Shutting down application...")


# Initialize the FastAPI application with lifespan management
app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text sentiment as positive, negative, or neutral",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API to verify it is running.
    
    Returns:
        dict: Status information with "status": "ok"
    
    Example:
        GET /health
        Response: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict sentiment of the provided text.
    
    Classifies text as positive, negative, or neutral using the pre-trained
    TF-IDF + Logistic Regression model.
    
    Args:
        request (PredictionRequest): Contains the text field with review to analyze
    
    Returns:
        PredictionResponse: Contains the original text, predicted sentiment, and confidence
    
    Raises:
        HTTPException: 500 if model is not loaded
    
    Example:
        POST /predict
        {
            "text": "I absolutely love this product!"
        }
        
        Response:
        {
            "text": "I absolutely love this product!",
            "sentiment": "positive",
            "confidence": 0.94
        }
    """
    try:
        # Get sentiment prediction and confidence score
        sentiment, confidence = sentiment_model.predict_sentiment(request.text)
        
        # Return structured response
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence
        )
    except RuntimeError as e:
        # Model not loaded
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # Invalid input
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment of multiple texts in batch.
    
    Efficient endpoint for processing multiple reviews at once.
    Maintains input order in the response.
    
    Args:
        request (BatchPredictionRequest): Contains list of texts to analyze
    
    Returns:
        BatchPredictionResponse: Contains list of predictions and total count
    
    Raises:
        HTTPException: 500 if model is not loaded, 400 for invalid input
    
    Example:
        POST /predict/batch
        {
            "texts": [
                "I love this!",
                "Terrible product.",
                "It's okay."
            ]
        }
        
        Response:
        {
            "results": [
                {"text": "I love this!", "sentiment": "positive", "confidence": 0.95},
                {"text": "Terrible product.", "sentiment": "negative", "confidence": 0.92},
                {"text": "It's okay.", "sentiment": "neutral", "confidence": 0.58}
            ],
            "count": 3
        }
    """
    try:
        # Process all texts and collect predictions
        predictions = []
        for text in request.texts:
            sentiment, confidence = sentiment_model.predict_sentiment(text)
            prediction = PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence
            )
            predictions.append(prediction)
        
        # Return batch response with results and count
        return BatchPredictionResponse(
            results=predictions,
            count=len(predictions)
        )
    except RuntimeError as e:
        # Model not loaded
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        # Invalid input
        raise HTTPException(status_code=400, detail=str(e))
