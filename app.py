
from pathlib import Path
import joblib
import streamlit as st

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# -----------------------------
# 2. Load the trained model
# -----------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "sentiment_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# 3. Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
    }

    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #172554, #1D4ED8);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #DBEAFE;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-top: 1rem;
    }

    .positive {
        color: #15803D;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .negative {
        color: #B91C1C;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .small-note {
        color: #64748B;
        font-size: 0.85rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        min-height: 3rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# 4. Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 Movie Sentiment Analyzer</h1>
        <p>
            Analyze movie reviews using Machine Learning.
            Discover whether a review expresses positive
            or negative sentiment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Analyze a movie review")
st.write(
    "Enter an English movie review below and let the model "
    "predict its sentiment."
)

# -----------------------------
# 5. Review input
# -----------------------------
review = st.text_area(
    "Your movie review",
    placeholder=(
        "Example: The story was engaging, "
        "the acting was excellent, and I loved the movie."
    ),
    height=170,
    max_chars=10000,
)

st.caption(f"Characters entered: {len(review)}")

analyze = st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True,
)

# -----------------------------
# 6. Prediction
# -----------------------------
if analyze:
    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        try:
            model = load_model()

            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]

            classes = list(model.classes_)
            confidence = float(
                probabilities[classes.index(prediction)]
            )

            st.markdown("---")
            st.markdown("### Analysis Result")

            if prediction == "pos":
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="positive">
                            😊 Positive Sentiment
                        </div>
                        <p>
                            The model predicts that this review
                            expresses a positive opinion.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            elif prediction == "neg":
                st.markdown(
                    """
                    <div class="result-card">
                        <div class="negative">
                            😞 Negative Sentiment
                        </div>
                        <p>
                            The model predicts that this review
                            expresses a negative opinion.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.write(f"Predicted class: {prediction}")

            st.markdown("#### Model probability")
            st.progress(confidence)
            st.write(f"**{confidence:.2%}**")

            st.caption(
                "This percentage is the model's estimated "
                "probability for its predicted class. "
                "It does not guarantee that the prediction "
                "is correct."
            )

            with st.expander("About this analysis"):
                st.write(
                    "The model uses TF-IDF text features and "
                    "a trained machine learning classifier. "
                    "Its performance may vary with review "
                    "length, wording, sarcasm, and unfamiliar "
                    "expressions."
                )

        except Exception as error:
            st.error("An error occurred while analyzing the review.")
            st.caption(str(error))

# -----------------------------
# 7. Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <p class="small-note" style="text-align:center;">
        Machine Learning Project · NLP · Sentiment Analysis
    </p>
    """,
    unsafe_allow_html=True,
)
