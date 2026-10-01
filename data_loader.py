"""
Data loading and preprocessing for Reuters newswire classification.
"""
import numpy as np
import pickle
import gzip
import os
from pathlib import Path
from urllib.request import urlretrieve
import torch
from torch.utils.data import Dataset, DataLoader

from config import NUM_WORDS, NUM_CLASSES, VAL_SPLIT, RANDOM_SEED, PROJECT_ROOT


# Reuters dataset URL (hosted by Keras)
REUTERS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters.npz"
REUTERS_WORD_INDEX_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters_word_index.json"
DATA_DIR = PROJECT_ROOT / "data"


def _download_reuters_data():
    """Download Reuters dataset if not present."""
    DATA_DIR.mkdir(exist_ok=True)
    
    npz_path = DATA_DIR / "reuters.npz"
    word_index_path = DATA_DIR / "reuters_word_index.json"
    
    if not npz_path.exists():
        print("Downloading Reuters dataset...")
        urlretrieve(REUTERS_URL, npz_path)
        print("Done!")
    
    if not word_index_path.exists():
        print("Downloading word index...")
        urlretrieve(REUTERS_WORD_INDEX_URL, word_index_path)
        print("Done!")
    
    return npz_path, word_index_path


def load_reuters_raw():
    """
    Load raw Reuters dataset.
    
    Returns:
        Tuple of (train_data, train_labels), (test_data, test_labels), word_index
    """
    import json
    
    npz_path, word_index_path = _download_reuters_data()
    
    # Load data - the file contains combined 'x' and 'y' arrays
    with np.load(npz_path, allow_pickle=True) as f:
        x_all = f["x"]
        y_all = f["y"]
    
    # Split into train/test (8982 train, 2246 test - standard Reuters split)
    train_size = 8982
    x_train = x_all[:train_size]
    y_train = y_all[:train_size]
    x_test = x_all[train_size:]
    y_test = y_all[train_size:]
    
    # Filter to num_words
    def filter_words(sequences, max_words):
        result = []
        for seq in sequences:
            filtered = [idx if idx < max_words else 2 for idx in seq]  # 2 = unknown
            result.append(filtered)
        return result
    
    x_train = filter_words(x_train, NUM_WORDS)
    x_test = filter_words(x_test, NUM_WORDS)
    
    # Load word index
    with open(word_index_path, "r") as f:
        word_index = json.load(f)
    
    return (x_train, y_train), (x_test, y_test), word_index


def load_reuters_data():
    """
    Load and preprocess the Reuters dataset.
    
    Returns:
        dict with keys: x_train, y_train, x_val, y_val, x_test, y_test,
                       train_labels_int, val_labels_int, test_labels_int,
                       word_index
    """
    # Load raw data
    (train_data, train_labels), (test_data, test_labels), word_index = load_reuters_raw()
    
    # Multi-hot encode inputs
    x_train_full = multi_hot_encode(train_data, NUM_WORDS)
    x_test = multi_hot_encode(test_data, NUM_WORDS)
    
    # One-hot encode labels
    y_train_full = one_hot_encode(train_labels, NUM_CLASSES)
    y_test = one_hot_encode(test_labels, NUM_CLASSES)
    
    # Split validation set
    x_val = x_train_full[:VAL_SPLIT]
    x_train = x_train_full[VAL_SPLIT:]
    y_val = y_train_full[:VAL_SPLIT]
    y_train = y_train_full[VAL_SPLIT:]
    
    # Keep integer labels for some metrics
    val_labels_int = train_labels[:VAL_SPLIT]
    train_labels_int = train_labels[VAL_SPLIT:]
    test_labels_int = test_labels
    
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_labels_int": train_labels_int,
        "val_labels_int": val_labels_int,
        "test_labels_int": test_labels_int,
        "word_index": word_index,
        "train_data_raw": list(train_data[VAL_SPLIT:]),  # For sequence models
        "val_data_raw": list(train_data[:VAL_SPLIT]),
        "test_data_raw": list(test_data),
    }


def multi_hot_encode(sequences, num_classes):
    """
    Multi-hot encode sequences of word indices.
    
    Args:
        sequences: List of lists of word indices
        num_classes: Size of vocabulary
        
    Returns:
        numpy array of shape (len(sequences), num_classes)
    """
    results = np.zeros((len(sequences), num_classes), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for idx in seq:
            if idx < num_classes:
                results[i, idx] = 1.0
    return results


def one_hot_encode(labels, num_classes):
    """
    One-hot encode integer labels.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
        
    Returns:
        numpy array of shape (len(labels), num_classes)
    """
    results = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results


def decode_newswire(sequence, word_index):
    """
    Decode a sequence of word indices back to text.
    
    Args:
        sequence: List of word indices
        word_index: Dictionary mapping words to indices
        
    Returns:
        Decoded string
    """
    reverse_word_index = {v: k for k, v in word_index.items()}
    # Raw reuters.npz indices match word_index directly (Keras only adds the +3 offset in its own loader)
    decoded = " ".join(
        reverse_word_index.get(i, "?") for i in sequence
    )
    return decoded


def get_topic_names():
    """
    Get human-readable topic names for Reuters dataset.
    Note: The actual topic names aren't readily available in Keras,
    so we use topic indices. In production, you'd map these to real names.
    
    Returns:
        List of topic name strings
    """
    # Common Reuters topic mappings (partial list)
    topic_names = {
        0: "cocoa", 1: "grain", 2: "veg-oil", 3: "earn", 4: "acq",
        5: "wheat", 6: "copper", 7: "housing", 8: "money-supply", 9: "coffee",
        10: "sugar", 11: "trade", 12: "reserves", 13: "ship", 14: "cotton",
        15: "carcass", 16: "crude", 17: "nat-gas", 18: "cpi", 19: "money-fx",
        20: "interest", 21: "gnp", 22: "meal-feed", 23: "alum", 24: "oilseed",
        25: "gold", 26: "tin", 27: "strategic-metal", 28: "livestock", 29: "retail",
        30: "ipi", 31: "iron-steel", 32: "rubber", 33: "heat", 34: "jobs",
        35: "lei", 36: "bop", 37: "zinc", 38: "orange", 39: "pet-chem",
        40: "dlr", 41: "gas", 42: "silver", 43: "wpi", 44: "hog", 45: "lead"
    }
    return [topic_names.get(i, f"topic_{i}") for i in range(NUM_CLASSES)]


class ReutersDataset(Dataset):
    """PyTorch Dataset for Reuters multi-hot encoded data."""
    
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ReutersSequenceDataset(Dataset):
    """PyTorch Dataset for Reuters sequential data (for embedding models)."""
    
    def __init__(self, sequences, labels, max_len=200):
        self.max_len = max_len
        self.sequences = self._pad_sequences(sequences)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def _pad_sequences(self, sequences):
        """Pad or truncate sequences to fixed length."""
        padded = np.zeros((len(sequences), self.max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_len)
            padded[i, :length] = seq[:length]
        return torch.tensor(padded, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def get_dataloaders(data_dict, batch_size=512, model_type="dense"):
    """
    Create PyTorch DataLoaders from data dictionary.
    
    Args:
        data_dict: Output from load_reuters_data()
        batch_size: Batch size for training
        model_type: "dense" for multi-hot, "sequence" for sequential
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if model_type == "sequence":
        train_dataset = ReutersSequenceDataset(
            data_dict["train_data_raw"], data_dict["y_train"]
        )
        val_dataset = ReutersSequenceDataset(
            data_dict["val_data_raw"], data_dict["y_val"]
        )
        test_dataset = ReutersSequenceDataset(
            data_dict["test_data_raw"], data_dict["y_test"]
        )
    else:
        train_dataset = ReutersDataset(data_dict["x_train"], data_dict["y_train"])
        val_dataset = ReutersDataset(data_dict["x_val"], data_dict["y_val"])
        test_dataset = ReutersDataset(data_dict["x_test"], data_dict["y_test"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    print("Loading Reuters data...")
    data = load_reuters_data()
    print(f"Training samples: {len(data['x_train'])}")
    print(f"Validation samples: {len(data['x_val'])}")
    print(f"Test samples: {len(data['x_test'])}")
    print(f"Input dimension: {data['x_train'].shape[1]}")
    print(f"Number of classes: {data['y_train'].shape[1]}")
    
    # Decode a sample
    sample_idx = 10
    decoded = decode_newswire(data['train_data_raw'][sample_idx], data['word_index'])
    print(f"\nSample newswire: {decoded[:200]}...")
    print(f"Label: {np.argmax(data['y_train'][sample_idx])}")
    
    # Test DataLoader
    train_loader, _, _ = get_dataloaders(data, batch_size=32)
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes: x={x_batch.shape}, y={y_batch.shape}")
