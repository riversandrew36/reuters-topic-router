"""
Calibration analysis for probabilistic predictions.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from pathlib import Path

import sys
sys.path.append('..')
from config import OUTPUTS_DIR


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data.
    
    Args:
        y_true: True labels (integer indices)
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (mean_predicted_probs, fraction_positives, bin_counts)
    """
    # Get predicted class and its probability
    y_pred = y_proba.argmax(axis=1)
    confidence = y_proba.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    
    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mean_predicted = np.zeros(n_bins)
    fraction_positive = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_predicted[i] = confidence[mask].mean()
            fraction_positive[i] = correct[mask].mean()
            bin_counts[i] = np.sum(mask)
    
    return mean_predicted, fraction_positive, bin_counts


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    mean_pred, frac_pos, counts = compute_calibration_curve(y_true, y_proba, n_bins)
    
    n_samples = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        if counts[i] > 0:
            ece += (counts[i] / n_samples) * np.abs(frac_pos[i] - mean_pred[i])
    
    return ece


def compute_mce(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE value
    """
    mean_pred, frac_pos, counts = compute_calibration_curve(y_true, y_proba, n_bins)
    
    valid_bins = counts > 0
    if not np.any(valid_bins):
        return 0.0
    
    return np.max(np.abs(frac_pos[valid_bins] - mean_pred[valid_bins]))


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[Path] = None,
    title: str = "Reliability Diagram"
):
    """
    Plot reliability diagram (calibration curve) with confidence histogram.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        figsize: Figure size
        save_path: Path to save figure
        title: Plot title
    """
    mean_pred, frac_pos, counts = compute_calibration_curve(y_true, y_proba, n_bins)
    ece = compute_ece(y_true, y_proba, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Reliability diagram
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    valid_bins = counts > 0
    
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(
        bin_centers[valid_bins], 
        frac_pos[valid_bins], 
        width=0.08, 
        alpha=0.7, 
        color="steelblue",
        edgecolor="black",
        label="Model"
    )
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives (Accuracy)")
    ax1.set_title(f"{title}\nECE = {ece:.3f}")
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal")
    
    # Confidence histogram
    confidence = y_proba.max(axis=1)
    ax2.hist(confidence, bins=n_bins, range=(0, 1), 
             alpha=0.7, color="steelblue", edgecolor="black")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reliability diagram to {save_path}")
    
    plt.show()
    return fig


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute all calibration metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dict with ECE, MCE, and other metrics
    """
    confidence = y_proba.max(axis=1)
    
    return {
        "ece": compute_ece(y_true, y_proba, n_bins),
        "mce": compute_mce(y_true, y_proba, n_bins),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "std_confidence": float(confidence.std()),
    }


def print_calibration_summary(metrics: Dict[str, float]):
    """Print formatted calibration metrics."""
    print("\n" + "="*50)
    print("CALIBRATION METRICS")
    print("="*50)
    
    print(f"\nExpected Calibration Error (ECE): {metrics['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE):  {metrics['mce']:.4f}")
    print(f"\nConfidence Statistics:")
    print(f"  Mean:   {metrics['mean_confidence']:.4f}")
    print(f"  Median: {metrics['median_confidence']:.4f}")
    print(f"  Std:    {metrics['std_confidence']:.4f}")
    
    # Interpretation
    if metrics['ece'] < 0.05:
        print("\n✓ Model is well-calibrated (ECE < 0.05)")
    elif metrics['ece'] < 0.15:
        print("\n⚠ Model has moderate calibration error (0.05 < ECE < 0.15)")
    else:
        print("\n✗ Model is poorly calibrated (ECE > 0.15)")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 10
    
    # Create somewhat well-calibrated predictions
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Create probabilities with varying confidence
    y_proba = np.random.dirichlet(np.ones(n_classes) * 0.5, n_samples)
    
    # Make it somewhat calibrated by boosting correct class
    for i in range(n_samples):
        y_proba[i, y_true[i]] += np.random.uniform(0.2, 0.5)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Compute and print metrics
    metrics = compute_calibration_metrics(y_true, y_proba)
    print_calibration_summary(metrics)
    
    # Plot
    plot_reliability_diagram(
        y_true, y_proba,
        save_path=OUTPUTS_DIR / "test_calibration.png"
    )
