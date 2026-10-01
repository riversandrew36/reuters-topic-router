"""
TF-IDF + Logistic Regression baseline model.
Strong, fast, and interpretable.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

import sys
sys.path.append('..')
from config import TFIDF_CONFIG, MODELS_DIR, NUM_CLASSES


class TFIDFLinearModel:
    """
    TF-IDF + Logistic Regression for text classification.
    
    This model:
    1. Converts raw text to TF-IDF features
    2. Uses multinomial logistic regression for classification
    3. Provides interpretability via feature weights
    """
    
    def __init__(
        self,
        max_features=None,
        ngram_range=None,
        C=None
    ):
        max_features = max_features or TFIDF_CONFIG["max_features"]
        ngram_range = ngram_range or TFIDF_CONFIG["ngram_range"]
        C = C or TFIDF_CONFIG["C"]
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )
        
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs"
        )
        
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier)
        ])
        
        self.is_fitted = False
        
    def fit(self, texts, labels):
        """
        Fit the TF-IDF + LogReg pipeline.
        
        Args:
            texts: List of text strings
            labels: Array of integer labels
        """
        print("Fitting TF-IDF + Logistic Regression...")
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        print("Done!")
        
    def predict(self, texts):
        """
        Predict class labels.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of predicted labels
        """
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts):
        """
        Predict class probabilities.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        return self.pipeline.predict_proba(texts)
    
    def get_top_words_per_class(self, n_words=10, topic_names=None):
        """
        Get top words for each class based on logistic regression coefficients.
        
        Args:
            n_words: Number of top words to return per class
            topic_names: Optional list of topic names
            
        Returns:
            Dict mapping class index/name to list of (word, weight) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.classifier.coef_
        
        result = {}
        n_classes = coefficients.shape[0]
        
        for class_idx in range(n_classes):
            class_weights = coefficients[class_idx]
            top_indices = np.argsort(class_weights)[-n_words:][::-1]
            
            top_words = [
                (feature_names[i], class_weights[i]) 
                for i in top_indices
            ]
            
            class_name = topic_names[class_idx] if topic_names else str(class_idx)
            result[class_name] = top_words
        
        return result
    
    def evaluate(self, texts, labels):
        """
        Evaluate model on test data.
        
        Args:
            texts: List of text strings
            labels: True labels
            
        Returns:
            Dict with accuracy and classification report
        """
        predictions = self.predict(texts)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            "accuracy": acc,
            "classification_report": report
        }
    
    def save(self, path=None):
        """Save model to disk."""
        path = path or MODELS_DIR / "tfidf_linear.joblib"
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")
        
    def load(self, path=None):
        """Load model from disk."""
        path = path or MODELS_DIR / "tfidf_linear.joblib"
        self.pipeline = joblib.load(path)
        self.vectorizer = self.pipeline.named_steps["tfidf"]
        self.classifier = self.pipeline.named_steps["clf"]
        self.is_fitted = True
        print(f"Model loaded from {path}")


def prepare_texts_from_sequences(sequences, word_index):
    """
    Convert sequences of word indices back to text for TF-IDF.
    
    Args:
        sequences: List of lists of word indices
        word_index: Dict mapping words to indices
        
    Returns:
        List of text strings
    """
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    texts = []
    for seq in sequences:
        # Raw reuters.npz indices match word_index directly (Keras only adds the +3 offset in its own loader)
        words = [reverse_word_index.get(i, "") for i in seq]
        texts.append(" ".join(w for w in words if w))
    
    return texts


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "..")
    from data_loader import load_reuters_data, get_topic_names
    
    # Load data
    data = load_reuters_data()
    
    # Prepare texts
    train_texts = prepare_texts_from_sequences(
        data["train_data_raw"], data["word_index"]
    )
    test_texts = prepare_texts_from_sequences(
        data["test_data_raw"], data["word_index"]
    )
    
    print(f"Sample text: {train_texts[0][:200]}...")
    
    # Train model
    model = TFIDFLinearModel()
    model.fit(train_texts, data["train_labels_int"])
    
    # Evaluate
    results = model.evaluate(test_texts, data["test_labels_int"])
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    
    # Show top words for a few classes
    topic_names = get_topic_names()
    top_words = model.get_top_words_per_class(n_words=5, topic_names=topic_names)
    
    print("\nTop words per class:")
    for topic in ["earn", "acq", "crude", "money-fx", "grain"]:
        if topic in top_words:
            words = [f"{w[0]}({w[1]:.2f})" for w in top_words[topic]]
            print(f"  {topic}: {', '.join(words)}")
