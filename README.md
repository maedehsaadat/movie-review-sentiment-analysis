# Movie Review Sentiment Analysis

A machine learning project that classifies movie reviews as positive or negative using two datasets and compares traditional text classification models.

## Project Overview

This project explores sentiment analysis using TF-IDF text features and two machine learning algorithms:

- Logistic Regression
- Multinomial Naive Bayes

Experiments were conducted on the NLTK Movie Reviews dataset and the IMDb movie reviews dataset.

A Streamlit application is included for interactive sentiment prediction.

## Technologies

- Python
- Scikit-learn
- NLTK
- Hugging Face Datasets
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Streamlit
- Pandas
- Matplotlib

## Datasets

### NLTK Movie Reviews

- 2,000 movie reviews
- 1,000 negative and 1,000 positive reviews
- 80/20 stratified train-test split

### IMDb Movie Reviews

- 25,000 training reviews
- 25,000 test reviews
- Balanced positive and negative classes
- Official train/test split

## Methodology

1. Load and prepare the review text.
2. Convert text into numerical features using TF-IDF.
3. Train Logistic Regression and Multinomial Naive Bayes.
4. Evaluate both models using classification metrics.
5. Compare model performance.
6. Use Streamlit to predict sentiment for new reviews.

## Results

### NLTK Dataset

| Model | Accuracy |
|---|---:|
| Logistic Regression | 84.25% |
| Multinomial Naive Bayes | 82.50% |

### IMDb Dataset

| Model | Accuracy |
|---|---:|
| Logistic Regression | 88.82% |
| Multinomial Naive Bayes | 85.65% |

Logistic Regression achieved higher accuracy than Multinomial Naive Bayes on both datasets.

Results from the two datasets should not be interpreted as a direct, controlled comparison because their sizes and data distributions differ.

## Run the IMDb App

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run imdb_app.py
```

The saved model file must be available at the path expected by the application. See the project setup instructions before running the app.

## Project Structure

```text
movie-review-sentiment-analysis/
├── sentiment_analysis.py
├── imdb_experiment.py
├── app.py
├── imdb_app.py
├── results/
├── results_imdb/
├── README.md
├── requirements.txt
└── .gitignore
```

## Future Improvements

- Add more detailed error analysis.
- Evaluate probability calibration.
- Experiment with additional text classification methods.
- Improve the interactive application.

## Author

Developed as a practical machine learning portfolio project.
