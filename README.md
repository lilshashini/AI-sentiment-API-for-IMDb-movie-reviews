# AI Sentiment API

A FastAPI-based sentiment analysis service that classifies text as positive, negative, or neutral with confidence scores.

## Quick Start

### Prerequisites

- **Python 3.9+** (verified on 3.9)
- **pip** (Python package manager)

### Setup Instructions

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd ai-sentiment-api
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   python train.py
   ```
   This will load the IMDB dataset, train a TF-IDF + Logistic Regression model, and save it to `model/sentiment_model.pkl`. You should see output like:
   ```
   Loading dataset...
   Training TF-IDF and Logistic Regression model...
   Evaluating model...
   
   --- Evaluation Report ---
   Accuracy:  0.8740
   Precision: 0.8751
   Recall:    0.8740
   F1 Score:  0.8739
   
   Model saved successfully to model/sentiment_model.pkl
   ```

5. **Start the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`

## Using the API

### Interactive API Documentation

Visit `http://127.0.0.1:8000/docs` in your browser to explore the API with Swagger UI.

### Health Check

Verify the service is running:

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{"status": "ok"}
```

### Make a Prediction

**Using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product, it exceeded all my expectations!"}'
```

Response:
```json
{
  "text": "I absolutely love this product, it exceeded all my expectations!",
  "sentiment": "positive",
  "confidence": 0.94
}
```

**Using Python:**
```python
import requests
import json

url = "http://127.0.0.1:8000/predict"
payload = {"text": "This is the worst experience I've ever had."}
response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

Response:
```json
{
  "text": "This is the worst experience I've ever had.",
  "sentiment": "negative",
  "confidence": 0.89
}
```

## Sentiment Classification Logic

The model returns sentiments based on the following thresholds:

- **Positive:** Probability of positive sentiment ≥ 0.60
- **Negative:** Probability of positive sentiment ≤ 0.40
- **Neutral:** Probability between 0.40 and 0.60

## Project Structure

```
ai-sentiment-api/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.py                       # Model training script
├── data/
│   └── IMDB Dataset.csv          # Training data
├── model/
│   └── sentiment_model.pkl       # Trained model (generated after running train.py)
└── app/
    ├── main.py                   # FastAPI application
    └── schemas.py                # Pydantic request/response models
```

## Approach

I chose a **TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer paired with Logistic Regression** for this sentiment classification task. This classical approach is lightweight, interpretable, and achieves 87.4% accuracy on the IMDB dataset—making it ideal for a production API where inference speed and explainability matter. The model was trained on a balanced dataset, providing a solid baseline for text sentiment analysis.

If I had more time, I would: (1) implement a neural network-based approach (e.g., fine-tuned BERT embeddings) to capture more nuanced linguistic patterns and improve accuracy, (2) add model versioning and monitoring to track performance drift in production, (3) expand the dataset with domain-specific reviews to improve generalization beyond movie reviews, and (4) implement caching and batch prediction endpoints to optimize throughput for high-volume API calls.

## Notes

- The model is loaded into memory on startup for low-latency predictions.
- Ensure the `model/sentiment_model.pkl` file exists before starting the API (run `python train.py` first).
- The API uses Pydantic for request/response validation, ensuring type safety.

---

**Happy sentiment analyzing!** 🚀
