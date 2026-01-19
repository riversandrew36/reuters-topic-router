import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from pathlib import Path

import sys
sys.path.append('..')
from config import OUTPUTS_DIR


def compute_selective_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    thresholds = np.linspace(0, 1, n_thresholds)
    accuracies = np.zeros(n_thresholds)
    coverages = np.zeros(n_thresholds)
    
    correct = (y_pred == y_true)
    n_total = len(y_true)
    
    for i, thresh in enumerate(thresholds):
        mask = confidence >= thresh
        coverage = np.sum(mask) / n_total
        coverages[i] = coverage
        
        if np.sum(mask) > 0:
            accuracies[i] = np.mean(correct[mask])
        else:
            accuracies[i] = np.nan
    
    return thresholds, accuracies, coverages


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    target_accuracy: float = 0.90,
    min_coverage: float = 0.01
) -> Dict:
    
    thresholds, accuracies, coverages = compute_selective_prediction_curve(
        y_true, y_pred, confidence
    )
    
    # Find thresholds that achieve target accuracy with valid coverage
    valid_mask = (accuracies >= target_accuracy) & (coverages >= min_coverage) & ~np.isnan(accuracies)
    
    if not np.any(valid_mask):
        # Return the threshold that gets closest to target
        valid_idx = ~np.isnan(accuracies) & (coverages >= min_coverage)
        if not np.any(valid_idx):
            return {"threshold": 1.0, "accuracy": np.nan, "coverage": 0.0, "achieved_target": False}
        
        best_idx = np.argmin(np.abs(accuracies[valid_idx] - target_accuracy))
        idx = np.where(valid_idx)[0][best_idx]
    else:
        # Find the threshold with maximum coverage among those achieving target
        idx = np.where(valid_mask)[0][np.argmax(coverages[valid_mask])]
    
    return {
        "threshold": thresholds[idx],
        "accuracy": accuracies[idx],
        "coverage": coverages[idx],
        "achieved_target": accuracies[idx] >= target_accuracy
    }


def compute_abstain_stats(
    confidence: np.ndarray,
    threshold: float
) -> Dict:
    """
    Compute statistics about abstention at a given threshold.
    
    Args:
        confidence: Confidence scores
        threshold: Confidence threshold
        
    Returns:
        Dict with abstention statistics
    """
    n_total = len(confidence)
    n_abstain = np.sum(confidence < threshold)
    n_predict = n_total - n_abstain
    
    return {
        "total_samples": n_total,
        "predictions": n_predict,
        "abstentions": n_abstain,
        "abstain_rate": n_abstain / n_total,
        "coverage": n_predict / n_total,
    }


def plot_accuracy_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    target_accuracies: List[float] = [0.85, 0.90, 0.95],
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[Path] = None,
    title: str = "Selective Prediction: Accuracy vs Coverage"
):
    
    thresholds, accuracies, coverages = compute_selective_prediction_curve(
        y_true, y_pred, confidence
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Accuracy vs Coverage
    valid = ~np.isnan(accuracies)
    ax1.plot(coverages[valid], accuracies[valid], "b-", linewidth=2, label="Model")
    
    # Mark target accuracy points
    colors = ["green", "orange", "red"]
    for i, target_acc in enumerate(target_accuracies):
        opt = find_optimal_threshold(y_true, y_pred, confidence, target_accuracy=target_acc)
        if opt["achieved_target"]:
            ax1.axhline(y=target_acc, color=colors[i % len(colors)], linestyle="--", alpha=0.5)
            ax1.scatter([opt["coverage"]], [opt["accuracy"]], 
                       color=colors[i % len(colors)], s=100, zorder=5,
                       label=f"{target_acc*100:.0f}% acc @ {opt['coverage']*100:.0f}% cov")
    
    # Baseline accuracy (all predictions)
    baseline_acc = np.mean(y_pred == y_true)
    ax1.axhline(y=baseline_acc, color="gray", linestyle=":", alpha=0.7, 
               label=f"Baseline: {baseline_acc*100:.1f}%")
    
    ax1.set_xlabel("Coverage (fraction of predictions made)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy vs Threshold
    ax2.plot(thresholds[valid], accuracies[valid], "b-", linewidth=2, label="Accuracy")
    ax2.plot(thresholds, coverages, "g--", linewidth=2, label="Coverage")
    
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Accuracy / Coverage")
    ax2.set_title("Accuracy and Coverage vs Threshold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved accuracy-coverage curve to {save_path}")
    
    plt.show()
    return fig


def print_selective_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    target_accuracies: List[float] = [0.85, 0.90, 0.95]
):
    """Print selective prediction analysis summary."""
    print("\n" + "="*50)
    print("SELECTIVE PREDICTION ANALYSIS")
    print("="*50)
    
    baseline_acc = np.mean(y_pred == y_true)
    print(f"\nBaseline Accuracy (all predictions): {baseline_acc*100:.1f}%")
    
    print(f"\nOptimal Thresholds for Target Accuracies:")
    print("-" * 50)
    
    for target_acc in target_accuracies:
        opt = find_optimal_threshold(y_true, y_pred, confidence, target_accuracy=target_acc)
        
        if opt["achieved_target"]:
            status = "✓"
            abstain_rate = 1 - opt["coverage"]
            print(f"{status} Target {target_acc*100:.0f}%: threshold={opt['threshold']:.3f}, "
                  f"accuracy={opt['accuracy']*100:.1f}%, coverage={opt['coverage']*100:.1f}%, "
                  f"abstain={abstain_rate*100:.1f}%")
        else:
            status = "✗"
            print(f"{status} Target {target_acc*100:.0f}%: not achievable "
                  f"(best: {opt['accuracy']*100:.1f}% @ {opt['coverage']*100:.1f}% coverage)")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 10
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    
    # Add some errors (more errors at lower confidence)
    confidence = np.random.beta(5, 2, n_samples)  # Skewed towards high confidence
    
    # Make errors inversely proportional to confidence
    error_prob = 1 - confidence
    errors = np.random.random(n_samples) < error_prob * 0.5
    y_pred[errors] = (y_pred[errors] + np.random.randint(1, n_classes, np.sum(errors))) % n_classes
    
    # Print summary
    print_selective_prediction_summary(y_true, y_pred, confidence)
    
    # Plot
    plot_accuracy_coverage_curve(
        y_true, y_pred, confidence,
        save_path=OUTPUTS_DIR / "test_selective_prediction.png"
    )
