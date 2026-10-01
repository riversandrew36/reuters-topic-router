import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append('..')
from config import OUTPUTS_DIR


def compute_input_gradients(
    model,
    x: torch.Tensor,
    target_class: Optional[int] = None,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute gradients of output w.r.t. input features.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (batch, input_dim)
        target_class: Class to compute gradients for (default: predicted class)
        device: Device to run on
        
    Returns:
        Gradients of shape (batch, input_dim)
    """
    model = model.to(device)
    model.eval()
    
    x = x.to(device)
    x.requires_grad_(True)
    
    # Forward pass
    logits = model.forward_logits(x)
    
    if target_class is None:
        target_class = logits.argmax(dim=1)
    elif isinstance(target_class, int):
        target_class = torch.tensor([target_class] * x.shape[0], device=device)
    
    # Compute gradients for the target class
    batch_size = x.shape[0]
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1)
    
    logits.backward(gradient=one_hot)
    
    gradients = x.grad.detach().cpu().numpy()
    return gradients


def compute_saliency_scores(
    gradients: np.ndarray,
    method: str = "absolute"
) -> np.ndarray:
    """
    Convert gradients to saliency scores.
    
    Args:
        gradients: Raw gradients from compute_input_gradients
        method: "absolute" or "signed"
        
    Returns:
        Saliency scores
    """
    if method == "absolute":
        return np.abs(gradients)
    elif method == "signed":
        return gradients
    else:
        raise ValueError(f"Unknown method: {method}")


def get_top_salient_features(
    saliency: np.ndarray,
    n_features: int = 10,
    word_index: Optional[Dict] = None
) -> List[List[Tuple]]:
    """
    Get top salient features for each sample.
    
    Args:
        saliency: Saliency scores of shape (batch, input_dim)
        n_features: Number of top features to return
        word_index: Optional word index for feature names
        
    Returns:
        List of lists of (feature_idx, saliency_score, word) tuples
    """
    reverse_word_index = None
    if word_index:
        reverse_word_index = {v: k for k, v in word_index.items()}
    
    results = []
    
    for i in range(saliency.shape[0]):
        sample_saliency = saliency[i]
        top_indices = np.argsort(sample_saliency)[-n_features:][::-1]
        
        top_features = []
        for idx in top_indices:
            word = reverse_word_index.get(idx, f"[{idx}]") if reverse_word_index else f"[{idx}]"
            top_features.append((idx, sample_saliency[idx], word))
        
        results.append(top_features)
    
    return results


def highlight_salient_words(
    sequence: List[int],
    saliency: np.ndarray,
    word_index: Dict,
    top_n: int = 10
) -> str:
    """
    Create a highlighted string showing salient words.
    
    Args:
        sequence: List of word indices
        saliency: Saliency scores for this sample
        word_index: Word to index mapping
        top_n: Number of top words to highlight
        
    Returns:
        String with salient words marked with **
    """
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    # Get saliency for words in sequence
    word_saliency = []
    for idx in sequence:
        word = reverse_word_index.get(idx, "?")
        sal = saliency[idx] if idx < len(saliency) else 0
        word_saliency.append((word, sal))
    
    # Find threshold for top N
    all_saliencies = [ws[1] for ws in word_saliency]
    if len(all_saliencies) > 0:
        threshold = sorted(all_saliencies, reverse=True)[min(top_n-1, len(all_saliencies)-1)]
    else:
        threshold = 0
    
    # Build highlighted string
    words = []
    for word, sal in word_saliency:
        if sal >= threshold and sal > 0:
            words.append(f"**{word}**")
        else:
            words.append(word)
    
    return " ".join(words)


def compute_integrated_gradients(
    model,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 50,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (batch, input_dim)
        baseline: Baseline input (default: zeros)
        target_class: Class to compute gradients for
        n_steps: Number of interpolation steps
        device: Device to run on
        
    Returns:
        Integrated gradients of shape (batch, input_dim)
    """
    model = model.to(device)
    model.eval()
    
    x = x.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    
    integrated_grads = torch.zeros_like(x)
    
    for alpha in alphas[1:]:  # Skip baseline (alpha=0)
        interpolated = baseline + alpha * (x - baseline)
        interpolated.requires_grad_(True)
        
        logits = model.forward_logits(interpolated)
        
        if target_class is None:
            target = logits.argmax(dim=1)
        else:
            target = torch.tensor([target_class] * x.shape[0], device=device)
        
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        logits.backward(gradient=one_hot)
        
        integrated_grads += interpolated.grad.detach() / n_steps
        interpolated.grad.zero_()
    
    # Scale by (x - baseline)
    integrated_grads = integrated_grads * (x - baseline)
    
    return integrated_grads.detach().cpu().numpy()


def print_saliency_analysis(
    model,
    x: torch.Tensor,
    sequences: List[List[int]],
    word_index: Dict,
    y_true: Optional[np.ndarray] = None,
    n_samples: int = 3,
    n_words: int = 10,
    device: str = "cpu"
):
    """
    Print saliency analysis for sample inputs.
    
    Args:
        model: PyTorch model
        x: Input tensor
        sequences: Raw sequences for decoding
        word_index: Word index mapping
        y_true: True labels
        n_samples: Number of samples to analyze
        n_words: Number of salient words to highlight
        device: Device
    """
    print("\n" + "="*50)
    print("SALIENCY ANALYSIS (Gradient-based)")
    print("="*50)
    
    # Get predictions first
    model.eval()
    with torch.no_grad():
        probs = model(x.to(device)).cpu().numpy()
    
    predictions = probs.argmax(axis=1)
    
    # Compute saliency
    gradients = compute_input_gradients(model, x[:n_samples], device=device)
    saliency = compute_saliency_scores(gradients)
    
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    for i in range(min(n_samples, len(x))):
        print(f"\n--- Sample {i+1} ---")
        
        pred_class = predictions[i]
        pred_prob = probs[i, pred_class]
        
        if y_true is not None:
            true_class = y_true[i] if isinstance(y_true[i], (int, np.integer)) else y_true[i].argmax()
            print(f"True: {true_class}, Predicted: {pred_class} ({pred_prob*100:.1f}%)")
        else:
            print(f"Predicted: {pred_class} ({pred_prob*100:.1f}%)")
        
        # Highlight salient words
        highlighted = highlight_salient_words(
            sequences[i], saliency[i], word_index, top_n=n_words
        )
        print(f"Text (top {n_words} words highlighted):")
        print(f"  {highlighted[:300]}...")


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "..")
    from data_loader import load_reuters_data, get_dataloaders
    from models.dense_nn import DenseNN
    
    device = "cpu"
    
    # Load data
    data = load_reuters_data()
    train_loader, val_loader, _ = get_dataloaders(data, batch_size=32)
    
    # Create untrained model for testing
    model = DenseNN()
    
    # Get a batch
    x_batch, y_batch = next(iter(val_loader))
    
    # Compute saliency
    print_saliency_analysis(
        model, x_batch,
        data["val_data_raw"][:5],
        data["word_index"],
        y_true=y_batch.argmax(dim=1).numpy(),
        n_samples=3,
        device=device
    )
