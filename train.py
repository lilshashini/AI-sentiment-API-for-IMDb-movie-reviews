import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path: str):
    """Loads the dataset and limits the size for faster local training."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # DECISION: Sampling 10,000 rows so it trains quickly on a local machine.
    # The full 50k dataset is great, but 10k is plenty for a strong baseline model.
    df = df.sample(n=10000, random_state=42)
    
    X = df['review']
    y = df['sentiment']
    return X, y

def train_model(X_train, y_train):
    """Builds and trains a scikit-learn pipeline."""
    print("Training TF-IDF and Logistic Regression model...")
    
    # DECISION: Using a Pipeline to bundle the vectorizer and classifier.
    # This ensures that when we load the model in FastAPI, it automatically
    # handles the text-to-vector transformation without extra code.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(random_state=42, max_iter=500))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Generates the required evaluation metrics."""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # DECISION: Using macro average since we are treating 'positive' and 'negative' equally.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("\n--- Evaluation Report ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-------------------------\n")

def save_model(model, output_path: str):
    """Saves the pipeline to disk as a .pkl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {output_path}")

if __name__ == "__main__":
    DATA_PATH = "data/IMDB Dataset.csv"
    MODEL_PATH = "model/sentiment_model.pkl"

    # 1. Load data
    X, y = load_data(DATA_PATH)
    
    # 2. Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train
    model = train_model(X_train, y_train)
    
    # 4. Evaluate (prints the report for your README)
    evaluate_model(model, X_test, y_test)
    
    # 5. Save to disk
    save_model(model, MODEL_PATH)