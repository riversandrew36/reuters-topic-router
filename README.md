# Reuters Topic Router

A portfolio-worthy multiclass classification system for Reuters newswire topic prediction with uncertainty quantification.

## Features

- **3 Model Architectures**: Dense NN (with MC Dropout), TF-IDF + Logistic Regression, Embedding + GlobalAveragePooling
- **Uncertainty Estimation**: Monte Carlo Dropout for predictive uncertainty
- **Abstain Option**: Configurable confidence thresholds for automatic vs manual routing
- **Comprehensive Evaluation**: Per-class P/R/F1, confusion matrix, calibration curves, selective prediction
- **Interpretability**: Top words per class (TF-IDF), gradient saliency (neural)
- **CLI + Web App**: Command-line tool and Streamlit interface

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train Models
```bash
# Train all models
python train.py

# Quick test (2 epochs, skip plots)
python train.py --quick-test

# Train specific models
python train.py --models dense tfidf
```

### CLI Prediction
```bash
# Single prediction
python predict_topic.py --text "Oil prices surge as OPEC cuts production"

# Interactive mode
python predict_topic.py

# Batch processing
python predict_topic.py --file articles.txt --output predictions.json
```

### Streamlit App
```bash
streamlit run app.py
```

## Project Structure

```
classification_project/
├── config.py                 # Configuration and hyperparameters
├── data_loader.py            # Data loading and preprocessing
├── train.py                  # Main training script
├── predict_topic.py          # CLI prediction tool
├── app.py                    # Streamlit web app
├── models/
│   ├── dense_nn.py           # Dense NN with MC Dropout
│   ├── tfidf_linear.py       # TF-IDF + LogReg
│   └── embedding_pooling.py  # Embedding sequence model
├── evaluation/
│   ├── metrics.py            # Per-class metrics
│   ├── confusion_analysis.py # Confusion matrix
│   ├── calibration.py        # Calibration curves
│   └── selective_prediction.py
├── interpretability/
│   ├── linear_weights.py     # Top words per class
│   └── gradient_saliency.py  # Neural saliency
├── outputs/                  # Generated plots and results
├── saved_models/             # Trained model weights
└── tests/
```

## MC Dropout Uncertainty

The Dense NN uses Monte Carlo Dropout for uncertainty estimation:
- Multiple forward passes (50 by default) with dropout enabled
- Computes predictive entropy and variance
- Enables selective prediction (abstain when uncertain)

```python
from models.dense_nn import MCDropoutPredictor

predictor = MCDropoutPredictor(model, n_samples=50)
result = predictor.predict(x)

print(f"Prediction: {result['predictions']}")
print(f"Confidence: {result['confidence']}")
print(f"Entropy: {result['entropy']}")
print(f"Should abstain: {predictor.should_abstain(result['entropy'], result['confidence'])}")
```

## Evaluation Metrics

After training, the following outputs are generated:
- `outputs/training_results.json` - Model comparison metrics
- `outputs/dense_calibration.png` - Calibration diagram
- `outputs/dense_selective_prediction.png` - Accuracy vs coverage curve
- `outputs/dense_confusion_matrix.png` - Confusion matrix
- `outputs/tfidf_top_words.png` - Top words per class
- `outputs/top_words_per_class.csv` - Exportable word importance

## Configuration

Edit `config.py` to adjust:
- Model hyperparameters (hidden dims, dropout rate, learning rate)
- MC Dropout settings (n_samples, entropy threshold, confidence threshold)
- Data parameters (vocabulary size, validation split)
