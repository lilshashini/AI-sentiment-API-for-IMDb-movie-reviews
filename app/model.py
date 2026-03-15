"""
Model loading and prediction logic for sentiment analysis.

This module handles:
- Loading the trained ML pipeline from disk
- Making predictions on new text input
- Computing confidence scores with threshold-based sentiment classification
"""

import pickle
import os
from typing import Tuple


# Global variable to store the loaded model (initialized on app startup)
_sentiment_model = None


def load_model(model_path: str = "model/sentiment_model.pkl"):
    """
    Load the pre-trained sentiment analysis model from disk.
    
    This function should be called once at application startup to avoid
    reloading the model on every prediction request.
    
    Args:
        model_path (str): Path to the serialized model pickle file.
                         Defaults to "model/sentiment_model.pkl"
    
    Returns:
        The loaded scikit-learn pipeline, or None if the file doesn't exist.
    
    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
    """
    global _sentiment_model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please run 'python train.py' first to train the model."
        )
    
    # Load the pickled scikit-learn pipeline
    # The pipeline includes both the TF-IDF vectorizer and the Logistic Regression classifier
    with open(model_path, 'rb') as f:
        _sentiment_model = pickle.load(f)
    
    print(f"✅ Model loaded successfully from {model_path}")
    return _sentiment_model


def get_model():
    """
    Retrieve the currently loaded sentiment model.
    
    Returns:
        The loaded scikit-learn pipeline, or None if not yet loaded.
    
    Raises:
        RuntimeError: If the model has not been loaded yet.
    """
    global _sentiment_model
    
    if _sentiment_model is None:
        raise RuntimeError(
            "Model has not been loaded. Call load_model() during application startup."
        )
    
    return _sentiment_model


def predict_sentiment(text: str) -> Tuple[str, float]:
    """
    Classify sentiment of the given text as positive, negative, or neutral.
    
    Uses probability thresholds for the predicted positive sentiment:
    - Positive: P(positive) > 0.60
    - Negative: P(positive) < 0.40
    - Neutral: 0.40 ≤ P(positive) ≤ 0.60 (ambiguous/borderline cases)
    
    The neutral zone (0.40-0.60) captures texts where the model is uncertain
    between positive and negative sentiment.
    
    Args:
        text (str): The input review text to classify.
    
    Returns:
        Tuple[str, float]: A tuple containing:
            - sentiment (str): Classification label ('positive', 'negative', or 'neutral')
            - confidence (float): Confidence score (0.00 to 1.00) rounded to 2 decimals
    
    Raises:
        RuntimeError: If the model has not been loaded yet.
        ValueError: If the input text is empty.
    """
    # Retrieve the loaded model
    model = get_model()
    
    # Validate input
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    
    # Get probability predictions from the model
    # predict_proba returns shape (1, 2) where:
    # - Column 0: probability of negative sentiment
    # - Column 1: probability of positive sentiment
    probabilities = model.predict_proba([text])[0]
    prob_negative = probabilities[0]  # P(negative)
    prob_positive = probabilities[1]  # P(positive)
    
    # Apply threshold-based classification logic
    # When model confidence is high on either side, classify accordingly
    # When model is uncertain (0.40-0.60), classify as neutral
    if prob_positive > 0.60:
        # Positive sentiment (confident prediction)
        sentiment = "positive"
        confidence = prob_positive
    elif prob_positive < 0.40:
        # Negative sentiment (confident prediction)
        sentiment = "negative"
        confidence = prob_negative
    else:
        # Neutral: Model is uncertain (0.40 <= P(positive) <= 0.60)
        # This captures borderline cases where the model cannot strongly
        # distinguish between positive and negative sentiment
        sentiment = "neutral"
        confidence = max(prob_negative, prob_positive)
    
    # Round confidence to 2 decimal places for readability
    rounded_confidence = round(float(confidence), 2)
    
    return sentiment, rounded_confidence
