# Core data science and ML libraries
import pandas as pd
import os
import pickle

# Scikit-learn imports for model training and evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to TF-IDF vectors
from sklearn.linear_model import LogisticRegression  # Binary classifier
from sklearn.pipeline import Pipeline  # Chains preprocessing with classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics

def load_data(file_path: str):
    """Loads the dataset and limits the size for faster local training."""
    print("Loading dataset...")
    # Read CSV file containing IMDB reviews and their sentiment labels (positive/negative)
    df = pd.read_csv(file_path)
    
    # DECISION: Sampling 10,000 rows so it trains quickly on a local machine.
    # The full 50k dataset is great, but 10k is plenty for a strong baseline model.
    # Using random_state=42 ensures reproducibility across runs
    df = df.sample(n=10000, random_state=42)
    
    # Extract text reviews and corresponding labels
    X = df['review']  # Features: raw review text
    y = df['sentiment']  # Labels: positive/negative classification
    return X, y

def train_model(X_train, y_train):
    """Builds and trains a scikit-learn pipeline."""
    print("Training TF-IDF and Logistic Regression model...")
    
    # DECISION: Using a Pipeline to bundle the vectorizer and classifier.
    # This ensures that when we load the model in FastAPI, it automatically
    # handles the text-to-vector transformation without extra code.
    
    # Pipeline steps:
    # 1. TfidfVectorizer: Converts raw text to numerical TF-IDF vectors
    #    - Removes common English stop words (the, a, an, etc.)
    #    - Limits to top 5000 features (most important unique words)
    # 2. LogisticRegression: Binary classifier trained on the vectorized text
    #    - random_state=42: Ensures reproducible results
    #    - max_iter=500: Maximum iterations for convergence
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(random_state=42, max_iter=500))
    ])
    
    # Fit the pipeline: vectorizes text and trains the classifier
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Generates the required evaluation metrics."""
    print("Evaluating model...")
    # Make predictions on the test set using the trained pipeline
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics to assess model performance
    # DECISION: Using macro average since we are treating 'positive' and 'negative' equally.
    
    # Accuracy: Percentage of correct predictions out of all predictions
    acc = accuracy_score(y_test, y_pred)
    
    # Precision: Of positive predictions, how many were actually correct?
    # (macro average treats both classes equally)
    prec = precision_score(y_test, y_pred, average='macro')
    
    # Recall: Of actual positive cases, how many did we correctly identify?
    # (macro average treats both classes equally)
    rec = recall_score(y_test, y_pred, average='macro')
    
    # F1 Score: Harmonic mean of precision and recall (balances both metrics)
    # (macro average treats both classes equally)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Display evaluation results
    print("\n--- Evaluation Report ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-------------------------\n")

def save_model(model, output_path: str):
    """Saves the pipeline to disk as a .pkl file."""
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Serialize the trained pipeline to a pickle file for later use in the API
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {output_path}")

if __name__ == "__main__":
    # Define file paths
    DATA_PATH = "data/IMDB Dataset.csv"  # Input: IMDB reviews dataset
    MODEL_PATH = "model/sentiment_model.pkl"  # Output: Trained model

    # Step 1: Load and preprocess data
    # Reads CSV, samples 10k rows for faster training, splits into features (X) and labels (y)
    X, y = load_data(DATA_PATH)
    
    # Step 2: Split into 80% training and 20% testing
    # Training set: Used to teach the model
    # Test set: Used to evaluate the model on unseen data
    # random_state=42: Ensures reproducible splits across runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Construct and train the ML pipeline
    # Vectorizes text with TF-IDF and trains a Logistic Regression classifier
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate on test set and print performance metrics
    # Shows accuracy, precision, recall, and F1 score
    evaluate_model(model, X_test, y_test)
    
    # Step 5: Persist the trained model to disk
    # The FastAPI server will load this pickle file on startup
    save_model(model, MODEL_PATH)