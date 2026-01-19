import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append('..')
from config import OUTPUTS_DIR


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: One of 'true', 'pred', 'all', or None
        
    Returns:
        Confusion matrix array
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    topic_names: Optional[List[str]] = None,
    normalize: str = "true",
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        topic_names: Optional list of class names
        normalize: Normalization method
        figsize: Figure size
        save_path: Path to save the figure
        title: Plot title
    """
    cm = compute_confusion_matrix(y_true, y_pred, normalize=normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # For large number of classes, we might want to sample or aggregate
    n_classes = cm.shape[0]
    
    if n_classes > 20:
        # Show top classes by support for clarity
        class_counts = np.bincount(y_true, minlength=n_classes)
        top_classes = np.argsort(class_counts)[-20:]
        
        cm_subset = cm[np.ix_(top_classes, top_classes)]
        if topic_names:
            labels = [topic_names[i] for i in top_classes]
        else:
            labels = [str(i) for i in top_classes]
        
        sns.heatmap(
            cm_subset, annot=False, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_title(f"{title} (Top 20 classes by frequency)")
    else:
        labels = topic_names if topic_names else [str(i) for i in range(n_classes)]
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_title(title)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()
    return fig


def get_top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 10,
    topic_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Get the most common confusion pairs.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n: Number of confusion pairs to return
        topic_names: Optional topic names
        
    Returns:
        List of dicts with confusion info
    """
    # Find misclassified samples
    misclassified = y_true != y_pred
    
    if not np.any(misclassified):
        return []
    
    true_misclassified = y_true[misclassified]
    pred_misclassified = y_pred[misclassified]
    
    # Count confusion pairs
    confusion_counts = {}
    for t, p in zip(true_misclassified, pred_misclassified):
        pair = (t, p)
        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    
    # Sort by count
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    result = []
    for (true_class, pred_class), count in sorted_pairs:
        true_name = topic_names[true_class] if topic_names else str(true_class)
        pred_name = topic_names[pred_class] if topic_names else str(pred_class)
        
        result.append({
            "true_class": true_class,
            "true_name": true_name,
            "pred_class": pred_class,
            "pred_name": pred_name,
            "count": count,
        })
    
    return result


def get_confusion_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: List[str],
    true_class: int,
    pred_class: int,
    n: int = 5
) -> List[str]:
    """
    Get example texts that were confused between two classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: List of text samples
        true_class: True class to filter by
        pred_class: Predicted class that was wrong
        n: Number of examples to return
        
    Returns:
        List of example text strings
    """
    mask = (y_true == true_class) & (y_pred == pred_class)
    indices = np.where(mask)[0][:n]
    
    return [texts[i] for i in indices]


def print_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: Optional[List[str]] = None,
    topic_names: Optional[List[str]] = None,
    n_confusions: int = 5,
    n_examples: int = 2
):
    """
    Print detailed error analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: Optional list of text samples
        topic_names: Optional topic names
        n_confusions: Number of top confusions to show
        n_examples: Number of examples per confusion
    """
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)
    
    n_total = len(y_true)
    n_correct = np.sum(y_true == y_pred)
    n_errors = n_total - n_correct
    
    print(f"\nTotal samples: {n_total}")
    print(f"Correct: {n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"Errors: {n_errors} ({100*n_errors/n_total:.1f}%)")
    
    top_confusions = get_top_confusions(y_true, y_pred, n=n_confusions, topic_names=topic_names)
    
    print(f"\nTop {n_confusions} Confusion Pairs:")
    print("-" * 50)
    
    for i, conf in enumerate(top_confusions, 1):
        print(f"\n{i}. {conf['true_name']} → {conf['pred_name']} ({conf['count']} errors)")
        
        if texts is not None and n_examples > 0:
            examples = get_confusion_examples(
                y_true, y_pred, texts,
                conf["true_class"], conf["pred_class"],
                n=n_examples
            )
            for j, ex in enumerate(examples):
                # Truncate long texts
                ex_short = ex[:150] + "..." if len(ex) > 150 else ex
                print(f"   Example {j+1}: \"{ex_short}\"")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 500
    n_classes = 10
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Add systematic confusions
    noise_idx = np.random.choice(n_samples, size=100, replace=False)
    y_pred[noise_idx] = (y_true[noise_idx] + 1) % n_classes
    
    topic_names = [f"topic_{i}" for i in range(n_classes)]
    texts = [f"Sample text for item {i}" for i in range(n_samples)]
    
    # Show top confusions
    print_error_analysis(y_true, y_pred, texts, topic_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, topic_names,
        save_path=OUTPUTS_DIR / "test_confusion_matrix.png"
    )
