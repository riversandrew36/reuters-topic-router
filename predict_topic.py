import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

from config import MODELS_DIR, MC_DROPOUT_CONFIG, NUM_WORDS
from data_loader import load_reuters_data, get_topic_names, multi_hot_encode
from models.dense_nn import DenseNN, MCDropoutPredictor


def encode_text(text: str, word_index: dict) -> np.ndarray:
    words = text.lower().split()
    indices = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3  
            if idx < NUM_WORDS:
                indices.append(idx)
    
    result = np.zeros((1, NUM_WORDS), dtype=np.float32)
    for idx in indices:
        result[0, idx] = 1.0
    
    return result


def predict_single(
    text: str,
    model: DenseNN,
    predictor: MCDropoutPredictor,
    word_index: dict,
    topic_names: list,
    topk: int = 5,
    threshold: float = None,
    verbose: bool = True
) -> dict:
    """Predict topic for a single text with uncertainty estimation."""
    threshold = threshold or MC_DROPOUT_CONFIG["confidence_threshold"]
    
    x = encode_text(text, word_index)
    x_tensor = torch.tensor(x)
    result = predictor.predict(x_tensor)
    
    probs = result["mean_probs"][0]
    predicted_class = result["predictions"][0]
    confidence = result["confidence"][0]
    entropy = result["entropy"][0]
    
    top_indices = np.argsort(probs)[-topk:][::-1]
    top_predictions = [
        {
            "topic": topic_names[idx],
            "topic_id": int(idx),
            "probability": float(probs[idx])
        }
        for idx in top_indices
    ]
    
    should_abstain = predictor.should_abstain(
        np.array([entropy]), np.array([confidence])
    )[0]
    
    if entropy < 0.5:
        uncertainty_level = "LOW"
    elif entropy < 1.5:
        uncertainty_level = "MEDIUM"
    else:
        uncertainty_level = "HIGH"
    
    output = {
        "input_text": text[:200] + ("..." if len(text) > 200 else ""),
        "predicted_topic": topic_names[predicted_class],
        "predicted_topic_id": int(predicted_class),
        "confidence": float(confidence),
        "entropy": float(entropy),
        "uncertainty_level": uncertainty_level,
        "should_abstain": bool(should_abstain),
        "decision": "ABSTAIN" if should_abstain else "ROUTE",
        "top_predictions": top_predictions,
    }
    
    if verbose:
        print_prediction(output)
    
    return output


def print_prediction(result: dict):
    """Pretty print a prediction result."""
    print("\n" + "="*60)
    print("TOPIC ROUTER PREDICTION")
    print("="*60)
    
    print(f"\nInput: \"{result['input_text']}\"")
    print(f"\n{'─'*60}")
    
    print(f"\nPredicted Topic: {result['predicted_topic'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Uncertainty: {result['uncertainty_level']} (entropy: {result['entropy']:.2f})")
    
    print(f"\nTop {len(result['top_predictions'])} Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        bar_len = int(pred['probability'] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {i}. {pred['topic']:<15} {bar} {pred['probability']*100:5.1f}%")
    
    print(f"\n{'─'*60}")
    if result["should_abstain"]:
        print("⚠️  Decision: ABSTAIN (confidence too low for automatic routing)")
    else:
        print(f"✓  Decision: ROUTE to '{result['predicted_topic']}' pipeline")
    print("="*60 + "\n")


def load_model_and_data():
    """Load trained model and necessary data."""
    # Load word index
    print("Loading word index...")
    data = load_reuters_data()
    word_index = data["word_index"]
    topic_names = get_topic_names()
    
    # Load model
    print("Loading model...")
    model = DenseNN()
    model_path = MODELS_DIR / "dense_nn.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python train.py --models dense' first.")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # Create MC Dropout predictor
    predictor = MCDropoutPredictor(model, n_samples=50, device="cpu")
    
    return model, predictor, word_index, topic_names


def main():
    parser = argparse.ArgumentParser(
        description="Predict Reuters topics with uncertainty estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_topic.py --text "Oil prices rise as OPEC cuts production"
  python predict_topic.py --text "Federal Reserve raises interest rates" --topk 3
  python predict_topic.py --text "Wheat harvest exceeds expectations" --threshold 0.8
  python predict_topic.py --file articles.txt --output results.json
        """
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to classify"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="File with texts to classify (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for predictions (JSON)"
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold for abstain decision"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        # Interactive mode
        print("No input provided. Entering interactive mode.")
        print("Type 'quit' to exit.\n")
        
        model, predictor, word_index, topic_names = load_model_and_data()
        
        while True:
            try:
                text = input("Enter text to classify: ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    break
                if text:
                    predict_single(
                        text, model, predictor, word_index, topic_names,
                        topk=args.topk, threshold=args.threshold
                    )
            except KeyboardInterrupt:
                print("\nExiting...")
                break
        
        return
    
    # Load model
    model, predictor, word_index, topic_names = load_model_and_data()
    
    results = []
    
    if args.text:
        result = predict_single(
            args.text, model, predictor, word_index, topic_names,
            topk=args.topk, threshold=args.threshold, verbose=not args.quiet
        )
        results.append(result)
    
    if args.file:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if not args.quiet:
                print(f"\n[{i+1}/{len(texts)}]")
            result = predict_single(
                text, model, predictor, word_index, topic_names,
                topk=args.topk, threshold=args.threshold, verbose=not args.quiet
            )
            results.append(result)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
