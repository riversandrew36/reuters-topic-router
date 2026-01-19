import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append('..')
from config import OUTPUTS_DIR


def get_top_words_per_class(
    vectorizer,
    classifier,
    n_words: int = 10,
    topic_names: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_
    
    result = {}
    n_classes = coefficients.shape[0]
    
    for class_idx in range(n_classes):
        class_weights = coefficients[class_idx]
        
        # Get top positive weights
        top_indices = np.argsort(class_weights)[-n_words:][::-1]
        
        top_words = [
            (feature_names[i], float(class_weights[i])) 
            for i in top_indices
        ]
        
        class_name = topic_names[class_idx] if topic_names else str(class_idx)
        result[class_name] = top_words
    
    return result


def save_top_words_to_csv(
    top_words: Dict[str, List[Tuple[str, float]]],
    save_path: Optional[Path] = None
):
   
    save_path = save_path or OUTPUTS_DIR / "top_words_per_class.csv"
    
    rows = []
    for class_name, words in top_words.items():
        for rank, (word, weight) in enumerate(words, 1):
            rows.append({
                "class": class_name,
                "rank": rank,
                "word": word,
                "weight": weight
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved top words to {save_path}")
    return df


def plot_top_words_for_class(
    top_words: Dict[str, List[Tuple[str, float]]],
    class_name: str,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
):
    
    if class_name not in top_words:
        print(f"Class '{class_name}' not found")
        return
    
    words_weights = top_words[class_name]
    words = [w[0] for w in words_weights][::-1]  # Reverse for horizontal bar
    weights = [w[1] for w in words_weights][::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(words)))
    ax.barh(words, weights, color=colors)
    
    ax.set_xlabel("Weight")
    ax.set_title(f"Top Words for Class: {class_name}")
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved word importance plot to {save_path}")
    
    plt.show()
    return fig


def plot_top_words_grid(
    top_words: Dict[str, List[Tuple[str, float]]],
    classes_to_show: Optional[List[str]] = None,
    n_words: int = 5,
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 3),
    save_path: Optional[Path] = None
):
    
    if classes_to_show is None:
        classes_to_show = list(top_words.keys())[:9]
    
    n_classes = len(classes_to_show)
    nrows = (n_classes + ncols - 1) // ncols
    
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_classes > 1 else [axes]
    
    for i, class_name in enumerate(classes_to_show):
        ax = axes[i]
        
        if class_name not in top_words:
            ax.set_visible(False)
            continue
        
        words_weights = top_words[class_name][:n_words]
        words = [w[0] for w in words_weights][::-1]
        weights = [w[1] for w in words_weights][::-1]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(words)))
        ax.barh(words, weights, color=colors)
        ax.set_title(class_name, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="x", alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Top Words per Class (TF-IDF Weights)", fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved word importance grid to {save_path}")
    
    plt.show()
    return fig


def print_top_words_summary(
    top_words: Dict[str, List[Tuple[str, float]]],
    classes_to_show: Optional[List[str]] = None,
    n_words: int = 5
):
    """Print top words for selected classes."""
    if classes_to_show is None:
        classes_to_show = list(top_words.keys())[:10]
    
    print("\n" + "="*50)
    print("TOP WORDS PER CLASS")
    print("="*50)
    
    for class_name in classes_to_show:
        if class_name not in top_words:
            continue
        
        words = top_words[class_name][:n_words]
        word_str = ", ".join([f"{w[0]}({w[1]:.2f})" for w in words])
        print(f"\n{class_name}:")
        print(f"  {word_str}")


if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    
    # Create synthetic texts
    texts = [
        "oil crude barrel price market",
        "gold silver metal precious",
        "dollar yen currency forex exchange",
        "earnings profit revenue stock",
        "wheat corn grain crop harvest",
    ] * 100
    
    labels = [0, 1, 2, 3, 4] * 100
    
    # Fit model
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(max_iter=100, multi_class="multinomial")
    classifier.fit(X, labels)
    
    # Get top words
    topic_names = ["crude", "gold", "forex", "earnings", "grain"]
    top_words = get_top_words_per_class(
        vectorizer, classifier, n_words=5, topic_names=topic_names
    )
    
    # Print and plot
    print_top_words_summary(top_words)
    plot_top_words_grid(top_words, save_path=OUTPUTS_DIR / "test_top_words.png")
