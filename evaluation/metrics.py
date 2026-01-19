
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)
from typing import Dict, List, Optional, Tuple, Any


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    topic_names: Optional[List[str]] = None,
    top_k_values: List[int] = [3, 5]
) -> Dict[str, Any]:
    results = {}
    
    # Basic accuracy
    results["accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    results["per_class"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }
    
    # Macro and weighted averages
    results["macro_precision"] = precision.mean()
    results["macro_recall"] = recall.mean()
    results["macro_f1"] = f1.mean()
    
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    results["weighted_precision"] = precision_w
    results["weighted_recall"] = recall_w
    results["weighted_f1"] = f1_w
    
    # Top-K accuracy (if probabilities provided)
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        labels = list(range(n_classes))
        for k in top_k_values:
            if k < n_classes:
                try:
                    results[f"top_{k}_accuracy"] = top_k_accuracy_score(
                        y_true, y_proba, k=k, labels=labels
                    )
                except ValueError:
                    # Handle edge cases
                    pass
    
    # Classification report as dict
    # Get unique labels that exist in the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Only pass topic_names if they match the number of unique labels
    if topic_names is not None:
        # Create filtered names for labels that exist
        filtered_names = [topic_names[i] if i < len(topic_names) else str(i) for i in unique_labels]
    else:
        filtered_names = None
    
    results["classification_report"] = classification_report(
        y_true, y_pred, 
        labels=unique_labels,
        target_names=filtered_names,
        output_dict=True,
        zero_division=0
    )
    
    return results


def get_worst_classes(
    metrics: Dict[str, Any],
    n: int = 10,
    by: str = "f1",
    topic_names: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Get the worst performing classes.
    
    Args:
        metrics: Output from compute_all_metrics
        n: Number of classes to return
        by: Metric to rank by ("precision", "recall", "f1")
        topic_names: Optional topic names
        
    Returns:
        List of (class_name, metric_value) tuples
    """
    per_class = metrics["per_class"]
    scores = per_class[by]
    support = per_class["support"]
    
    # Filter out classes with no support
    valid_indices = np.where(support > 0)[0]
    valid_scores = [(i, scores[i]) for i in valid_indices]
    
    # Sort by score (ascending for worst)
    sorted_scores = sorted(valid_scores, key=lambda x: x[1])[:n]
    
    result = []
    for idx, score in sorted_scores:
        name = topic_names[idx] if topic_names else str(idx)
        result.append((name, score))
    
    return result


def get_best_classes(
    metrics: Dict[str, Any],
    n: int = 10,
    by: str = "f1",
    topic_names: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Get the best performing classes.
    
    Args:
        metrics: Output from compute_all_metrics
        n: Number of classes to return
        by: Metric to rank by ("precision", "recall", "f1")
        topic_names: Optional topic names
        
    Returns:
        List of (class_name, metric_value) tuples
    """
    per_class = metrics["per_class"]
    scores = per_class[by]
    support = per_class["support"]
    
    valid_indices = np.where(support > 0)[0]
    valid_scores = [(i, scores[i]) for i in valid_indices]
    
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:n]
    
    result = []
    for idx, score in sorted_scores:
        name = topic_names[idx] if topic_names else str(idx)
        result.append((name, score))
    
    return result


def print_metrics_summary(metrics: Dict[str, Any], topic_names: Optional[List[str]] = None):
    """Print a formatted summary of metrics."""
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS SUMMARY")
    print("="*50)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nMacro-averaged:")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro_f1']:.4f}")
    
    print(f"\nWeighted-averaged:")
    print(f"  Precision: {metrics['weighted_precision']:.4f}")
    print(f"  Recall:    {metrics['weighted_recall']:.4f}")
    print(f"  F1-Score:  {metrics['weighted_f1']:.4f}")
    
    # Top-K accuracy
    for key in metrics:
        if key.startswith("top_"):
            print(f"  {key.replace('_', '-')}: {metrics[key]:.4f}")
    
    # Best and worst classes
    print(f"\nTop 5 Best Classes (by F1):")
    for name, score in get_best_classes(metrics, n=5, topic_names=topic_names):
        print(f"  {name}: {score:.4f}")
    
    print(f"\nTop 5 Worst Classes (by F1):")
    for name, score in get_worst_classes(metrics, n=5, topic_names=topic_names):
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)
    n_samples = 500
    n_classes = 46
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=100, replace=False)
    y_pred[noise_idx] = np.random.randint(0, n_classes, 100)
    
    # Random probabilities
    y_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    print_metrics_summary(metrics)
