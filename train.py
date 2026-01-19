import argparse
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from config import (
    DENSE_NN_CONFIG, EMBEDDING_CONFIG, TFIDF_CONFIG,
    MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
)
from data_loader import (
    load_reuters_data, get_dataloaders, get_topic_names
)
from models.dense_nn import DenseNN, train_dense_nn, MCDropoutPredictor
from models.tfidf_linear import TFIDFLinearModel, prepare_texts_from_sequences
from models.embedding_pooling import EmbeddingPoolingModel, train_embedding_model
from evaluation.metrics import compute_all_metrics, print_metrics_summary
from evaluation.confusion_analysis import plot_confusion_matrix, print_error_analysis
from evaluation.calibration import plot_reliability_diagram, compute_calibration_metrics, print_calibration_summary
from evaluation.selective_prediction import plot_accuracy_coverage_curve, print_selective_prediction_summary


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_all_models(args):
    set_seed(RANDOM_SEED)
    
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    data = load_reuters_data()
    topic_names = get_topic_names()
    
    print(f"Training samples: {len(data['x_train'])}")
    print(f"Validation samples: {len(data['x_val'])}")
    print(f"Test samples: {len(data['x_test'])}")
    
    results = {}
    
    # ==========================================
    # 1. TF-IDF + Logistic Regression
    # ==========================================
    if "tfidf" in args.models:
        print("\n" + "="*50)
        print("TRAINING: TF-IDF + Logistic Regression")
        print("="*50)
        
        # Prepare texts
        train_texts = prepare_texts_from_sequences(
            data["train_data_raw"], data["word_index"]
        )
        val_texts = prepare_texts_from_sequences(
            data["val_data_raw"], data["word_index"]
        )
        test_texts = prepare_texts_from_sequences(
            data["test_data_raw"], data["word_index"]
        )
        
        tfidf_model = TFIDFLinearModel()
        tfidf_model.fit(train_texts, data["train_labels_int"])
        tfidf_model.save()
        
        # Evaluate
        y_pred = tfidf_model.predict(test_texts)
        y_proba = tfidf_model.predict_proba(test_texts)
        
        metrics = compute_all_metrics(
            data["test_labels_int"], y_pred, y_proba, topic_names
        )
        print_metrics_summary(metrics, topic_names)
        
        results["tfidf"] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top_3_accuracy": metrics.get("top_3_accuracy"),
        }
        
        # Save interpretability
        from interpretability.linear_weights import (
            get_top_words_per_class, save_top_words_to_csv, plot_top_words_grid
        )
        
        top_words = get_top_words_per_class(
            tfidf_model.vectorizer, tfidf_model.classifier,
            n_words=10, topic_names=topic_names
        )
        save_top_words_to_csv(top_words)
        
        if not args.quick_test:
            plot_top_words_grid(
                top_words,
                classes_to_show=["earn", "acq", "crude", "money-fx", "grain", "trade"],
                save_path=OUTPUTS_DIR / "tfidf_top_words.png"
            )
    
    # ==========================================
    # 2. Dense NN with MC Dropout
    # ==========================================
    if "dense" in args.models:
        print("\n" + "="*50)
        print("TRAINING: Dense NN with MC Dropout")
        print("="*50)
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data, batch_size=DENSE_NN_CONFIG["batch_size"]
        )
        
        dense_model = DenseNN(dropout_rate=DENSE_NN_CONFIG["dropout_rate"])
        print(f"Model parameters: {sum(p.numel() for p in dense_model.parameters()):,}")
        
        epochs = args.epochs if args.epochs else DENSE_NN_CONFIG["epochs"]
        history = train_dense_nn(
            dense_model, train_loader, val_loader,
            epochs=epochs, device=device
        )
        
        # Save model
        torch.save(dense_model.state_dict(), MODELS_DIR / "dense_nn.pt")
        
        # Standard evaluation
        dense_model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for x_batch, _ in test_loader:
                probs = dense_model(x_batch.to(device)).cpu().numpy()
                all_probs.append(probs)
                all_preds.append(probs.argmax(axis=1))
        
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        
        metrics = compute_all_metrics(
            data["test_labels_int"], y_pred, y_proba, topic_names
        )
        print_metrics_summary(metrics, topic_names)
        
        # MC Dropout evaluation
        print("\n--- MC Dropout Uncertainty Estimation ---")
        predictor = MCDropoutPredictor(dense_model, n_samples=50, device=device)
        
        mc_preds = []
        mc_confidence = []
        mc_entropy = []
        
        for x_batch, _ in test_loader:
            result = predictor.predict(x_batch)
            mc_preds.extend(result["predictions"])
            mc_confidence.extend(result["confidence"])
            mc_entropy.extend(result["entropy"])
        
        mc_preds = np.array(mc_preds)
        mc_confidence = np.array(mc_confidence)
        mc_entropy = np.array(mc_entropy)
        
        # Selective prediction analysis
        print_selective_prediction_summary(
            data["test_labels_int"], mc_preds, mc_confidence
        )
        
        results["dense_nn"] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top_3_accuracy": metrics.get("top_3_accuracy"),
            "mean_entropy": float(mc_entropy.mean()),
            "mean_confidence": float(mc_confidence.mean()),
        }
        
        if not args.quick_test:
            # Calibration
            plot_reliability_diagram(
                data["test_labels_int"], y_proba,
                save_path=OUTPUTS_DIR / "dense_calibration.png"
            )
            
            # Selective prediction
            plot_accuracy_coverage_curve(
                data["test_labels_int"], mc_preds, mc_confidence,
                save_path=OUTPUTS_DIR / "dense_selective_prediction.png"
            )
            
            # Confusion matrix
            plot_confusion_matrix(
                data["test_labels_int"], y_pred, topic_names,
                save_path=OUTPUTS_DIR / "dense_confusion_matrix.png"
            )
    
    # ==========================================
    # 3. Embedding + Pooling Model
    # ==========================================
    if "embedding" in args.models:
        print("\n" + "="*50)
        print("TRAINING: Embedding + GlobalAveragePooling")
        print("="*50)
        
        train_loader, val_loader, test_loader = get_dataloaders(
            data, batch_size=EMBEDDING_CONFIG["batch_size"], model_type="sequence"
        )
        
        embed_model = EmbeddingPoolingModel()
        print(f"Model parameters: {sum(p.numel() for p in embed_model.parameters()):,}")
        
        epochs = args.epochs if args.epochs else EMBEDDING_CONFIG["epochs"]
        history = train_embedding_model(
            embed_model, train_loader, val_loader,
            epochs=epochs, device=device
        )
        
        # Save model
        torch.save(embed_model.state_dict(), MODELS_DIR / "embedding_pooling.pt")
        
        # Evaluate
        embed_model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for x_batch, _ in test_loader:
                probs = embed_model(x_batch.to(device)).cpu().numpy()
                all_probs.append(probs)
                all_preds.append(probs.argmax(axis=1))
        
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        
        metrics = compute_all_metrics(
            data["test_labels_int"], y_pred, y_proba, topic_names
        )
        print_metrics_summary(metrics, topic_names)
        
        results["embedding"] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "top_3_accuracy": metrics.get("top_3_accuracy"),
        }
        
        if not args.quick_test:
            plot_confusion_matrix(
                data["test_labels_int"], y_pred, topic_names,
                save_path=OUTPUTS_DIR / "embedding_confusion_matrix.png"
            )
    
    # ==========================================
    # Save Results Summary
    # ==========================================
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in model_results.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    # Save to JSON
    results["timestamp"] = datetime.now().isoformat()
    with open(OUTPUTS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUTS_DIR / 'training_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Train Reuters topic classification models")
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["tfidf", "dense", "embedding"],
        choices=["tfidf", "dense", "embedding"],
        help="Models to train"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default epochs"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (fewer epochs, skip plots)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage"
    )
    
    args = parser.parse_args()
    
    if args.quick_test and args.epochs is None:
        args.epochs = 2
    
    train_all_models(args)


if __name__ == "__main__":
    main()
