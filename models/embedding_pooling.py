"""
Embedding + GlobalAveragePooling model for sequence classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')
from config import NUM_WORDS, NUM_CLASSES, EMBEDDING_CONFIG


class EmbeddingPoolingModel(nn.Module):
    """
    Sequence model with learned embeddings.
    
    Architecture:
        Embedding(vocab_size, embed_dim) → GlobalAveragePooling → Dense(hidden) → Dense(num_classes)
    """
    
    def __init__(
        self,
        vocab_size=NUM_WORDS,
        embedding_dim=None,
        hidden_dim=None,
        num_classes=NUM_CLASSES,
        dropout_rate=None,
        max_len=200
    ):
        super().__init__()
        
        embedding_dim = embedding_dim or EMBEDDING_CONFIG["embedding_dim"]
        hidden_dim = hidden_dim or EMBEDDING_CONFIG["hidden_dim"]
        dropout_rate = dropout_rate or EMBEDDING_CONFIG["dropout_rate"]
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) with word indices
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Global average pooling over sequence dimension
        # Mask out padding (index 0)
        mask = (x != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        masked_embedded = embedded * mask
        
        # Sum and divide by non-padding length
        summed = masked_embedded.sum(dim=1)  # (batch, embed_dim)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled = summed / lengths  # (batch, embed_dim)
        
        # Dense layers
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return F.softmax(logits, dim=-1)
    
    def forward_logits(self, x):
        """Forward pass returning raw logits."""
        embedded = self.embedding(x)
        
        mask = (x != 0).unsqueeze(-1).float()
        masked_embedded = embedded * mask
        summed = masked_embedded.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
        
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_embedding_model(
    model,
    train_loader,
    val_loader,
    epochs=None,
    learning_rate=None,
    device="cpu",
    early_stopping_patience=5
):
    """
    Train the embedding pooling model.
    
    Args:
        model: EmbeddingPoolingModel instance
        train_loader: Training DataLoader (should use ReutersSequenceDataset)
        val_loader: Validation DataLoader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        early_stopping_patience: Epochs to wait before early stopping
        
    Returns:
        dict with training history
    """
    epochs = epochs or EMBEDDING_CONFIG["epochs"]
    learning_rate = learning_rate or EMBEDDING_CONFIG["learning_rate"]
    
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
    train_loader, val_loader, test_loader = get_dataloaders(
        data, batch_size=512, model_type="sequence"
    )
    
    # Create model
    model = EmbeddingPoolingModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quick training test
    history = train_embedding_model(
        model, train_loader, val_loader,
        epochs=5, device=device
    )
    
    # Test prediction
    model.eval()
    x_test, y_test = next(iter(test_loader))
    x_test = x_test.to(device)
    
    with torch.no_grad():
        probs = model(x_test[:5])
    
    print("\nSample predictions:")
    print(f"Shape: {probs.shape}")
    print(f"Predictions: {probs.argmax(dim=1).cpu().numpy()}")
    print(f"True labels: {y_test[:5].argmax(dim=1).numpy()}")
