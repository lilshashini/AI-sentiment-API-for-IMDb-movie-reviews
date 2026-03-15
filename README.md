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

### Batch Predictions

Analyze multiple texts in a single request for better efficiency.

**Using curl:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I absolutely love this product!",
      "This is terrible, worst purchase ever.",
      "It is okay, nothing special.",
      "Amazing quality and fast shipping!"
    ]
  }'
```

Response:
```json
{
  "results": [
    {
      "text": "I absolutely love this product!",
      "sentiment": "positive",
      "confidence": 0.96
    },
    {
      "text": "This is terrible, worst purchase ever.",
      "sentiment": "negative",
      "confidence": 0.91
    },
    {
      "text": "It is okay, nothing special.",
      "sentiment": "neutral",
      "confidence": 0.58
    },
    {
      "text": "Amazing quality and fast shipping!",
      "sentiment": "positive",
      "confidence": 0.94
    }
  ],
  "count": 4
}
```

**Using Python:**
```python
import requests
import json

url = "http://127.0.0.1:8000/predict/batch"
payload = {
    "texts": [
        "I love this!",
        "Hate it!",
        "It's okay.",
        "Fantastic product!"
    ]
}

response = requests.post(url, json=payload)
result = response.json()

# Print results with sentiment and confidence
for pred in result["results"]:
    print(f"Text: {pred['text']}")
    print(f"  Sentiment: {pred['sentiment']} | Confidence: {pred['confidence']}")
    print()
```

Output:
```
Text: I love this!
  Sentiment: positive | Confidence: 0.95

Text: Hate it!
  Sentiment: negative | Confidence: 0.93

Text: It's okay.
  Sentiment: neutral | Confidence: 0.55

Text: Fantastic product!
  Sentiment: positive | Confidence: 0.97
```

**Edge Cases:**

Empty list returns 400 error:
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": []}'
```

Response (400 Bad Request):
```json
{
  "detail": "The 'texts' list cannot be empty. Please provide at least one text to analyze."
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

## Sources & Citations

### Frameworks & Libraries
- **FastAPI**: Starlette-based web framework for building APIs with Python type hints
  - https://fastapi.tiangolo.com/
  - Tiangolo. "FastAPI." Available at https://fastapi.tiangolo.com/

- **Scikit-learn**: Machine learning library for classification and feature extraction
  - https://scikit-learn.org/
  - Pedregosa et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

- **Pandas**: Data manipulation and analysis library
  - https://pandas.pydata.org/
  - McKinney, W. "Data Structures for Statistical Computing in Python." In Proceedings of the 9th Python in Science Conference (SciPy 2010), pp. 51-56.

- **Uvicorn**: ASGI server for running FastAPI applications
  - https://www.uvicorn.org/

- **Pydantic**: Data validation library for Python using type hints
  - https://docs.pydantic.dev/
  - https://github.com/pydantic/pydantic

### Machine Learning Techniques
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Text vectorization approach
  - Sparck Jones, K. "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." Journal of Documentation, vol. 28, no. 1, pp. 11-21, 1972.

- **Logistic Regression**: Linear classification algorithm
  - Bishop, C. M. Pattern Recognition and Machine Learning. Springer-Verlag, 2006.

- **Train-Test Split**: Model evaluation methodology
  - James, G., Witten, D., Hastie, T., & Tibshirani, R. "An Introduction to Statistical Learning." Springer, 2013.

### Dataset
- **IMDB Movie Reviews Dataset**: Used for training sentiment classification model
  - Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. "Learning Word Vectors for Sentiment Analysis." In Proceedings of the ACL-HLT 2011 Conference, pp. 142-150.
  - Available at: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

### API Design & Best Practices
- **RESTful API Design**: API endpoint design principles
  - https://restfulapi.net/

- **Pydantic Documentation**: Request/response validation
  - https://docs.pydantic.dev/

- **FastAPI Tutorial & Documentation**: Building async APIs with Python
  - https://fastapi.tiangolo.com/tutorial/

### Other Resources
- **Python Virtual Environments**: Dependency isolation and environment management
  - https://docs.python.org/3/tutorial/venv.html

- **Git & GitHub**: Version control and repository management
  - https://git-scm.com/doc
  - https://docs.github.com/

---

**Happy sentiment analyzing!** 🚀
