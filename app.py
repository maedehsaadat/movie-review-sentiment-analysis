
from pathlib import Path
import html
import joblib
import streamlit as st


# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "imdb_sentiment_model.joblib"
)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            145deg, #171526 0%, #211d35 55%, #171526 100%
        );
        color: #F4F1FA;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label, span {
        color: #F4F1FA;
    }

    .hero {
        text-align: center;
        padding: 25px 15px 20px;
        border: 1px solid #39334F;
        border-radius: 20px;
        background: rgba(31, 28, 49, 0.85);
        margin-bottom: 25px;
    }

    .hero-icon {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .hero-title {
        color: #E7B85C;
        font-size: 35px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #C7C0D9;
        font-size: 16px;
        line-height: 1.7;
    }

    .model-tag {
        display: inline-block;
        margin-top: 12px;
        padding: 7px 13px;
        border-radius: 20px;
        background: #302944;
        color: #E7B85C;
        font-size: 13px;
    }

    /* Review input and placeholder */
    textarea {
        background-color: #211D35 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #51456C !important;
        border-radius: 12px !important;
        caret-color: #E7B85C !important;
    }

    textarea::placeholder {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    textarea:focus {
        border-color: #E7B85C !important;
        box-shadow: 0 0 0 1px #E7B85C !important;
    }

    div.stButton > button {
        width: 100%;
        border: none;
        border-radius: 11px;
        background: linear-gradient(90deg, #C99B45, #E7B85C);
        color: #211D35;
        font-size: 16px;
        font-weight: 700;
        padding: 12px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #F0CA79;
        color: #171526;
        border: none;
    }

    .result-card {
        background: #211D35;
        border: 1px solid #49405F;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .result-label {
        color: #BEB5D3;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .positive {
        color: #73D6A0;
        font-size: 27px;
        font-weight: 750;
    }

    .negative {
        color: #FF929E;
        font-size: 27px;
        font-weight: 750;
    }

    .confidence-number {
        color: #E7B85C;
        font-size: 32px;
        font-weight: 750;
        margin-top: 5px;
    }

    .footnote {
        color: #A9A1BE;
        font-size: 12px;
        text-align: center;
        line-height: 1.7;
        margin-top: 25px;
    }

    div[data-testid="stMetric"] {
        background: #211D35;
        border: 1px solid #49405F;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #C7C0D9;
    }

    div[data-testid="stMetricValue"] {
        color: #E7B85C;
    }

    /* Empty review warning - force black text */
    #empty-review-warning,
    #empty-review-warning * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }

    #empty-review-warning {
        background-color: #FFF3CD !important;
        border: 2px solid #E0B84F;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 15px;
        font-size: 16px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🎬</div>
        <div class="hero-title">IMDb Sentiment Analyzer</div>
        <div class="hero-subtitle">
            Discover the sentiment behind movie reviews
            using machine learning.
        </div>
        <div class="model-tag">
            TF-IDF &nbsp;·&nbsp; Logistic Regression
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Review input
# -----------------------------
st.markdown("### ✍️ Your Movie Review")

review = st.text_area(
    "Enter your review below",
    placeholder=(
        "The movie was captivating, the acting was excellent, "
        "and the story kept me interested until the end..."
    ),
    height=180,
    label_visibility="collapsed",
)

st.caption("Write your review in English for best results.")

analyze = st.button("✨  Analyze Sentiment", type="primary")


# -----------------------------
# Prediction
# -----------------------------
if analyze:

    if not review.strip():

        st.markdown(
            """
            <div id="empty-review-warning">
                ⚠️ Please enter a movie review first.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif not MODEL_PATH.exists():

        st.error(f"Model file not found: {MODEL_PATH}")

    else:

        try:
            model = load_model()
            prediction = model.predict([review])[0]

            st.markdown("---")
            st.markdown("### 📊 Analysis Result")

            if prediction == "pos":
                result_text = "😊 Positive Sentiment"
                result_class = "positive"

            elif prediction == "neg":
                result_text = "😞 Negative Sentiment"
                result_class = "negative"

            else:
                result_text = html.escape(str(prediction))
                result_class = "positive"

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted sentiment
                    </div>
                    <div class="{result_class}">
                        {result_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba([review])[0]
                classes = model.classes_

                probability_map = dict(
                    zip(classes, probabilities)
                )

                confidence = float(max(probabilities)) * 100

                positive_prob = (
                    float(probability_map.get("pos", 0)) * 100
                )

                negative_prob = (
                    float(probability_map.get("neg", 0)) * 100
                )

                st.markdown("#### Prediction Confidence")

                st.markdown(
                    f"""
                    <div class="confidence-number">
                        {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(confidence / 100)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Positive probability",
                        f"{positive_prob:.2f}%",
                    )

                with col2:
                    st.metric(
                        "Negative probability",
                        f"{negative_prob:.2f}%",
                    )

                st.caption(
                    "These values represent the model's estimated "
                    "class probabilities, not a guarantee of correctness."
                )

        except Exception as error:
            st.error(f"An error occurred: {error}")


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footnote">
        IMDb Movie Reviews Dataset<br>
        Built with Python · Streamlit · Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
