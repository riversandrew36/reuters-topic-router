"""
Dense Neural Network with MC Dropout for uncertainty estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('..')
from config import NUM_WORDS, NUM_CLASSES, DENSE_NN_CONFIG, MC_DROPOUT_CONFIG


class DenseNN(nn.Module):
    """
    Dense neural network for multi-class classification with MC Dropout.
    
    Architecture:
        Input(10000) → Dense(64) → ReLU → Dropout → Dense(64) → ReLU → Dropout → Dense(46) → Softmax
    """
    
    def __init__(
        self, 
        input_dim=NUM_WORDS, 
        hidden_dims=None, 
        num_classes=NUM_CLASSES,
        dropout_rate=0.3
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = DENSE_NN_CONFIG["hidden_dims"]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)
        
    def forward(self, x):
        """Forward pass (softmax applied for probabilities)."""
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)
    
    def forward_logits(self, x):
        """Forward pass returning raw logits (for loss computation)."""
        x = self.hidden_layers(x)
        return self.output_layer(x)


class MCDropoutPredictor:
    """
    Wrapper for MC Dropout inference.
    
    Runs multiple forward passes with dropout enabled to estimate
    predictive uncertainty.
    """
    
    def __init__(self, model, n_samples=None, device="cpu"):
        self.model = model
        self.n_samples = n_samples or MC_DROPOUT_CONFIG["n_samples"]
        self.device = device
        self.model.to(device)
        
    def predict(self, x, return_samples=False):
        """
        Make predictions with uncertainty estimation via MC Dropout.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_samples: If True, return all MC samples
            
        Returns:
            dict with keys:
                - mean_probs: Mean predicted probabilities (batch, num_classes)
                - predictions: Predicted class indices (batch,)
                - confidence: Max probability for each sample (batch,)
                - entropy: Predictive entropy (batch,)
                - variance: Mean variance across classes (batch,)
                - samples: All MC samples if return_samples=True (n_samples, batch, num_classes)
        """
        self.model.train()  # Keep dropout active
        
        x = x.to(self.device)
        samples = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                probs = self.model(x)
                samples.append(probs.cpu().numpy())
        
        samples = np.array(samples)  # (n_samples, batch, num_classes)
        
        # Compute statistics
        mean_probs = samples.mean(axis=0)  # (batch, num_classes)
        predictions = mean_probs.argmax(axis=1)  # (batch,)
        confidence = mean_probs.max(axis=1)  # (batch,)
        
        # Predictive entropy: -sum(p * log(p))
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        # Mean variance across classes
        variance = samples.var(axis=0).mean(axis=1)  # (batch,)
        
        result = {
            "mean_probs": mean_probs,
            "predictions": predictions,
            "confidence": confidence,
            "entropy": entropy,
            "variance": variance,
        }
        
        if return_samples:
            result["samples"] = samples
            
        return result
    
    def should_abstain(self, entropy, confidence):
        """
        Determine if model should abstain from prediction.
        
        Args:
            entropy: Predictive entropy values
            confidence: Max probability values
            
        Returns:
            Boolean array indicating abstention
        """
        entropy_threshold = MC_DROPOUT_CONFIG["entropy_threshold"]
        confidence_threshold = MC_DROPOUT_CONFIG["confidence_threshold"]
        
        return (entropy > entropy_threshold) | (confidence < confidence_threshold)


def train_dense_nn(
    model, 
    train_loader, 
    val_loader, 
    epochs=None, 
    learning_rate=None,
    device="cpu",
    early_stopping_patience=5
):
    """
    Train the Dense NN model.
    
    Args:
        model: DenseNN model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        early_stopping_patience: Epochs to wait before early stopping
        
    Returns:
        dict with training history
    """
    epochs = epochs or DENSE_NN_CONFIG["epochs"]
    learning_rate = learning_rate or DENSE_NN_CONFIG["learning_rate"]
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model.forward_logits(x_batch)
            
            # Convert one-hot to class indices for CrossEntropyLoss
            targets = y_batch.argmax(dim=1)
            loss = criterion(logits, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += x_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                logits = model.forward_logits(x_batch)
                targets = y_batch.argmax(dim=1)
                loss = criterion(logits, targets)
                
                val_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += x_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "..")
    from data_loader import load_reuters_data, get_dataloaders
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    data = load_reuters_data()
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=512)
    
    # Create and train model
    model = DenseNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_dense_nn(
        model, train_loader, val_loader, 
        epochs=5, device=device
    )
    
    # Test MC Dropout prediction
    predictor = MCDropoutPredictor(model, n_samples=20, device=device)
    x_test, y_test = next(iter(test_loader))
    result = predictor.predict(x_test[:5])
    
    print("\nMC Dropout predictions:")
    print(f"Predictions: {result['predictions']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Entropy: {result['entropy']}")
    print(f"Should abstain: {predictor.should_abstain(result['entropy'], result['confidence'])}")
