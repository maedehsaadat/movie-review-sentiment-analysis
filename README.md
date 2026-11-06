# IMDb Movie Review Sentiment Analysis

A machine learning project that analyzes movie reviews and predicts whether their sentiment is **Positive** or **Negative**.

## Live Demo

Try the deployed application:

[**IMDb Sentiment Analyzer – Streamlit App**](https://maedehsaadat-movie-review-sentiment-analysis-imdb-app-zmiwor.streamlit.app/)

## Project Overview

This project demonstrates a machine learning approach to binary sentiment classification. It uses text preprocessing and TF-IDF feature extraction to transform movie reviews into numerical features, then applies classification models to predict sentiment.

The project includes two sentiment analysis experiments using different datasets and a Streamlit web application that allows users to enter a movie review and receive a sentiment prediction with confidence scores.

## Project Structure

```text
movie-review-sentiment-analysis/
│
├── models/
│   ├── imdb_sentiment_model.joblib
│   ├── sentiment_model.joblib
│   └── .gitkeep
│
├── results/
│   └── NLTK experiment results
│
├── results_imdb/
│   └── IMDb experiment results
│
├── app.py
├── imdb_app.py
├── sentiment_analysis.py
├── imdb_experiment.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

### File Descriptions

| File / Directory | Description |
|---|---|
| `models/` | Stores trained machine learning models. |
| `imdb_sentiment_model.joblib` | Trained IMDb sentiment model used by the deployed application. |
| `results/` | Contains evaluation results from the NLTK experiment. |
| `results_imdb/` | Contains evaluation results from the IMDb experiment. |
| `app.py` | Streamlit application for the initial sentiment analysis experiment. |
| `imdb_app.py` | Streamlit application for IMDb sentiment prediction. |
| `sentiment_analysis.py` | Implements the initial sentiment analysis experiment using the NLTK movie reviews dataset. |
| `imdb_experiment.py` | Implements the IMDb sentiment analysis experiment. |
| `requirements.txt` | Lists the required Python dependencies. |
| `README.md` | Project documentation. |
| `.gitignore` | Specifies files and directories excluded from Git tracking. |

## Features

- Positive and negative sentiment classification
- Text preprocessing and TF-IDF feature extraction
- Machine learning models, including Logistic Regression and Multinomial Naive Bayes
- Experiments on NLTK Movie Reviews and IMDb datasets
- Model evaluation and performance comparison
- Confusion matrices and classification metrics
- Analysis of misclassified reviews
- Interactive web application built with Streamlit
- Prediction confidence and positive/negative probabilities
- Saved model artifacts for reuse

## Technologies

- Python
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Joblib
- Streamlit
- Git & GitHub

## Saved Model Artifacts

The trained IMDb model is stored in the `models/` directory.

| File | Description |
|---|---|
| `imdb_sentiment_model.joblib` | Trained IMDb sentiment model used by the deployed Streamlit application. |

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/maedehsaadat/movie-review-sentiment-analysis.git
```

### 2. Navigate to the project directory

```bash
cd movie-review-sentiment-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the IMDb Streamlit application

```bash
streamlit run imdb_app.py
```

Make sure the trained model is available at:

```text
models/imdb_sentiment_model.joblib
```

### Run the Initial Sentiment Analysis Application

To run the initial NLTK-based application:

```bash
streamlit run app.py
```

## How to Use

1. Open the Streamlit application.
2. Enter an English movie review in the text area.
3. Click the **Analyze Sentiment** button.
4. View the predicted sentiment and its confidence score.
5. Review the positive and negative probability values.

## Limitations

- The application performs binary sentiment classification (Positive or Negative).
- Prediction quality depends on the training data and the model's ability to generalize to new reviews.
- The model may not interpret sarcasm, irony, or complex context reliably.
- The application is primarily designed for English-language reviews.
