"""
Streamlit app for Reuters Topic Router.

Run with: streamlit run app.py
"""
import streamlit as st
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="Reuters Topic Router",
    page_icon="",
    layout="wide"
)

from config import MODELS_DIR, MC_DROPOUT_CONFIG, NUM_WORDS
from data_loader import load_reuters_data, get_topic_names
from models.dense_nn import DenseNN, MCDropoutPredictor
from models.tfidf_linear import TFIDFLinearModel
from predict_topic import encode_text


@st.cache_resource
def load_resources():
    """Load models and data (cached)."""
    data = load_reuters_data()
    word_index = data["word_index"]
    topic_names = get_topic_names()
    
    # Load Dense NN model
    dense_model = DenseNN()
    model_path = MODELS_DIR / "dense_nn.pt"
    dense_loaded = False
    if model_path.exists():
        dense_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        dense_loaded = True
    
    predictor = MCDropoutPredictor(dense_model, n_samples=50, device="cpu")
    
    # Load TF-IDF model
    tfidf_model = TFIDFLinearModel()
    tfidf_path = MODELS_DIR / "tfidf_linear.joblib"
    tfidf_loaded = False
    if tfidf_path.exists():
        tfidf_model.load()
        tfidf_loaded = True
    
    return {
        "dense_model": dense_model,
        "predictor": predictor,
        "tfidf_model": tfidf_model,
        "word_index": word_index,
        "topic_names": topic_names,
        "dense_loaded": dense_loaded,
        "tfidf_loaded": tfidf_loaded,
    }


def create_probability_chart(top_predictions, height=300):
    """Create horizontal bar chart for predictions."""
    topics = [p["topic"] for p in top_predictions][::-1]
    probs = [p["probability"] * 100 for p in top_predictions][::-1]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=topics,
        orientation='h',
        marker_color='steelblue',
        text=[f"{p:.1f}%" for p in probs],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top Predictions",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 100])
    )
    
    return fig


def create_uncertainty_gauge(entropy, confidence):
    """Create gauge chart for uncertainty."""
    # Normalize entropy to 0-100 scale (max entropy ~3.8 for 46 classes)
    uncertainty_pct = min(entropy / 3.8 * 100, 100)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=uncertainty_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Uncertainty Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
    st.title("Reuters Topic Router")
    st.markdown("""
    **AI-powered news article classification with uncertainty estimation.**
    
    This system classifies news articles into 46 topics and provides confidence scores
    with the option to abstain when uncertain.
    """)
    
    # Load resources
    resources = load_resources()
    
    if not resources["dense_loaded"] and not resources["tfidf_loaded"]:
        st.error("""
        **No models found!**
        
        Please train the models first by running:
        ```bash
        python train.py
        ```
        """)
        return
    
    # Sidebar settings
    st.sidebar.header("Settings")
    
    # Model selection
    model_options = []
    if resources["tfidf_loaded"]:
        model_options.append("TF-IDF + LogReg (faster, higher confidence)")
    if resources["dense_loaded"]:
        model_options.append("Dense NN + MC Dropout (uncertainty estimation)")
    
    selected_model = st.sidebar.radio("Model", model_options, index=0)
    use_tfidf = "TF-IDF" in selected_model
    
    topk = st.sidebar.slider("Top-K Predictions", 3, 10, 5)
    
    if not use_tfidf:
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0, MC_DROPOUT_CONFIG["confidence_threshold"],
            help="Abstain if confidence below this value"
        )
        
        entropy_threshold = st.sidebar.slider(
            "Entropy Threshold",
            0.0, 3.5, MC_DROPOUT_CONFIG["entropy_threshold"],
            help="Abstain if entropy above this value"
        )
        
        mc_samples = st.sidebar.slider(
            "MC Dropout Samples",
            10, 100, 50,
            help="More samples = better uncertainty estimates"
        )
        
        resources["predictor"].n_samples = mc_samples
    else:
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.20,
            help="Abstain if confidence below this value"
        )
    
    # Main input
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Text")
        
        # Example texts
        example_texts = {
            "Select an example...": "",
            "Corporate Earnings": """The company reported quarterly earnings of 50 cents per share 
compared with 35 cents a year ago. Revenue rose 15 percent to 2.3 billion dollars. 
The chairman said profits were driven by strong demand and cost cutting measures. 
Analysts had expected earnings of 45 cents per share.""",
            "Acquisition News": """Company announced it has agreed to acquire rival firm for 500 million dollars 
in cash and stock. The acquisition will create the largest manufacturer in the industry. 
The deal is expected to close by the end of the year pending regulatory approval. 
Shareholders will receive 25 dollars per share.""",
            "Crude Oil Market": """Oil prices rose sharply today after OPEC announced production cuts of 
one million barrels per day. Crude futures gained 3 percent to close at 65 dollars per barrel. 
The increase reflects concerns about supply shortages and rising global demand. 
Saudi Arabia led the production reduction agreement.""",
            "Currency Exchange": """The dollar fell against the yen and mark following the Federal Reserve 
interest rate decision. Currency traders sold dollars after the central bank signaled 
it would hold rates steady. The yen rose to 125 per dollar while the mark gained 
to 1.65 per dollar in active trading.""",
            "Grain & Agriculture": """Wheat futures rose on reports of drought conditions in major producing regions. 
The Agriculture Department forecast lower production estimates for the harvest season. 
Corn and soybean prices also gained on concerns about crop damage. Export demand 
remained strong from Japan and other Asian markets.""",
            "Trade & Economics": """The trade deficit widened to 15 billion dollars in the latest month as imports 
exceeded exports. Government officials said the gap reflects strong consumer demand 
for foreign goods. Trade negotiations with Japan continue over market access issues. 
The Commerce Department will release revised figures next week.""",
        }
        
        selected_example = st.selectbox("Try an example:", list(example_texts.keys()))
        
        if selected_example != "Select an example...":
            default_text = example_texts[selected_example]
        else:
            default_text = ""
        
        text_input = st.text_area(
            "Enter news text to classify:",
            value=default_text,
            height=150,
            placeholder="Type or paste a news article here..."
        )
        
        predict_button = st.button("Predict Topic", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("About")
        model_info = "**TF-IDF + LogReg**" if use_tfidf else "**Dense NN + MC Dropout**"
        st.markdown(f"""
        **Current Model:** {model_info}
        
        **TF-IDF** is faster and produces 
        higher confidence on short text.
        
        **Dense NN** provides uncertainty 
        estimation via MC Dropout.
        
        **Topics include:**
        earn, acq, crude, money-fx, 
        grain, trade, interest, wheat...
        """)
    
    # Prediction
    if predict_button and text_input.strip():
        with st.spinner("Analyzing..."):
            topic_names = resources["topic_names"]
            
            if use_tfidf:
                # TF-IDF prediction
                tfidf_model = resources["tfidf_model"]
                probs = tfidf_model.predict_proba([text_input])[0]
                predicted_class = probs.argmax()
                confidence = probs.max()
                entropy = None
                should_abstain = confidence < confidence_threshold
            else:
                # Dense NN + MC Dropout prediction
                x = encode_text(text_input, resources["word_index"])
                x_tensor = torch.tensor(x)
                
                result = resources["predictor"].predict(x_tensor)
                
                probs = result["mean_probs"][0]
                predicted_class = result["predictions"][0]
                confidence = result["confidence"][0]
                entropy = result["entropy"][0]
                
                should_abstain = (
                    entropy > entropy_threshold or 
                    confidence < confidence_threshold
                )
            
            # Get top-k
            top_indices = np.argsort(probs)[-topk:][::-1]
            top_predictions = [
                {
                    "topic": topic_names[idx],
                    "topic_id": int(idx),
                    "probability": float(probs[idx])
                }
                for idx in top_indices
            ]
        
        st.markdown("---")
        st.subheader("Results")
        
        # Result columns
        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
        
        with res_col1:
            if should_abstain:
                st.error(f"""
                ### ABSTAIN
                *Confidence too low for automatic routing*
                """)
            else:
                st.success(f"""
                ### ROUTE
                **{topic_names[predicted_class].upper()}**
                """)
        
        with res_col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        with res_col3:
            if entropy is not None:
                if entropy < 0.5:
                    st.metric("Uncertainty", "LOW", delta=f"entropy: {entropy:.2f}")
                elif entropy < 1.5:
                    st.metric("Uncertainty", "MEDIUM", delta=f"entropy: {entropy:.2f}")
                else:
                    st.metric("Uncertainty", "HIGH", delta=f"entropy: {entropy:.2f}")
            else:
                st.metric("Model", "TF-IDF", delta="No uncertainty")
        
        # Charts
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            fig_probs = create_probability_chart(top_predictions)
            st.plotly_chart(fig_probs, use_container_width=True)
        
        with chart_col2:
            if entropy is not None:
                fig_gauge = create_uncertainty_gauge(entropy, confidence)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.info("**TF-IDF model** doesn't provide uncertainty estimation. Use Dense NN for MC Dropout uncertainty.")
        
        # Details expander
        with st.expander("Detailed Results"):
            result_dict = {
                "model": "TF-IDF + LogReg" if use_tfidf else "Dense NN + MC Dropout",
                "predicted_topic": topic_names[predicted_class],
                "confidence": float(confidence),
                "should_abstain": bool(should_abstain),
                "top_predictions": top_predictions,
            }
            if entropy is not None:
                result_dict["entropy"] = float(entropy)
            st.json(result_dict)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8em;">
    Reuters Topic Router | MC Dropout Uncertainty Estimation | Portfolio Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

