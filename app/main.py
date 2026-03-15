from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pickle
import os
import sklearn  # Required for unpickling scikit-learn models
from app.schemas import PredictionRequest, PredictionResponse
from typing import List
from app.schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest

# Create a global dictionary to store the loaded model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the ML model when the server starts, and cleans up when it shuts down."""
    model_path = "model/sentiment_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            ml_models["sentiment_model"] = pickle.load(f)
        print("✅ Model loaded successfully on startup!")
    else:
        print(f"❌ Error: Model not found at {model_path}")
        
    yield # The app runs during this yield
    
    # Cleanup on shutdown
    ml_models.clear()

# Initialize the FastAPI app
app = FastAPI(lifespan=lifespan, title="Sentiment Analysis API")

@app.get("/health")
def health_check():
    """Endpoint to verify the service is running."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """Accepts text and returns predicted sentiment and confidence score."""
    model = ml_models.get("sentiment_model")
    
    if not model:
        raise HTTPException(status_code=500, detail="Machine learning model is not loaded.")

    # 1. Get the probability scores from scikit-learn
    # predict_proba returns an array like [[prob_negative, prob_positive]]
    probabilities = model.predict_proba([request.text])[0]
    prob_negative = probabilities[0]
    prob_positive = probabilities[1]

    # 2. Apply our threshold logic to create a 3-class system
    if prob_positive >= 0.60:
        sentiment = "positive"
        confidence = prob_positive
    elif prob_positive <= 0.40:
        sentiment = "negative"
        confidence = prob_negative
    else:
        sentiment = "neutral"
        # For neutral, confidence is whichever probability was slightly higher
        confidence = max(prob_negative, prob_positive)

    # 3. Format the confidence score to 2 decimal places (e.g., 0.94)
    rounded_confidence = round(float(confidence), 2)

    return PredictionResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=rounded_confidence
    )
    
@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch_sentiment(request: BatchPredictionRequest):
    """Accepts a list of texts and returns predictions for all of them."""
    
    # 1. Handle the edge case: empty list (Returns 400 error)
    if len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="The texts list cannot be empty.")
        
    model = ml_models.get("sentiment_model")
    if not model:
        raise HTTPException(status_code=500, detail="Machine learning model is not loaded.")

    # 2. Get probabilities for the entire list at once (highly efficient!)
    probabilities_list = model.predict_proba(request.texts)
    
    responses = []
    
    # 3. Loop through each text and its corresponding probabilities
    for i, text in enumerate(request.texts):
        prob_negative = probabilities_list[i][0]
        prob_positive = probabilities_list[i][1]
        
        # Apply the exact same threshold logic for 3 classes
        if prob_positive >= 0.60:
            sentiment = "positive"
            confidence = prob_positive
        elif prob_positive <= 0.40:
            sentiment = "negative"
            confidence = prob_negative
        else:
            sentiment = "neutral"
            confidence = max(prob_negative, prob_positive)
            
        # Append the structured response object
        responses.append(
            PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=round(float(confidence), 2)
            )
        )
        
    return responses