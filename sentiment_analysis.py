
import joblib
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Create output directories
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Download dataset if needed
nltk.download("movie_reviews", quiet=True)

# 3. Load dataset
texts = []
labels = []

for file_id in movie_reviews.fileids():
    texts.append(" ".join(movie_reviews.words(file_id)))
    labels.append(movie_reviews.categories(file_id)[0])

df = pd.DataFrame({
    "text": texts,
    "sentiment": labels
})

print("Dataset size:", len(df))
print("\nClass distribution:")
print(df["sentiment"].value_counts())

df["sentiment"].value_counts().rename_axis(
    "sentiment"
).reset_index(name="count").to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

all_results = []
trained_models = {}

# 6. Train and evaluate models
for name, classifier in models.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

    print("\n" + "=" * 60)
    print("Training:", name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["neg", "pos"],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    trained_models[name] = model

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["neg", "pos"],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    filename = name.lower().replace(" ", "_")

    report_df.to_csv(
        OUTPUT_DIR / f"{filename}_classification_report.csv"
    )

    # Save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["neg", "pos"],
        display_labels=["Negative", "Positive"],
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{filename}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    # Save misclassified reviews
    errors = pd.DataFrame({
        "review": X_test,
        "actual_sentiment": y_test,
        "predicted_sentiment": predictions
    })

    errors = errors[
        errors["actual_sentiment"] != errors["predicted_sentiment"]
    ].copy()

    errors.to_csv(
        OUTPUT_DIR / f"{filename}_errors.csv",
        index=False,
        encoding="utf-8"
    )

    print("Misclassified reviews:", len(errors))

    # Collect evaluation metrics
    all_results.append({
        "model": name,
        "accuracy": accuracy,
        "negative_precision": report["Negative"]["precision"],
        "negative_recall": report["Negative"]["recall"],
        "negative_f1": report["Negative"]["f1-score"],
        "positive_precision": report["Positive"]["precision"],
        "positive_recall": report["Positive"]["recall"],
        "positive_f1": report["Positive"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "misclassified_reviews": len(errors)
    })

# 7. Compare models
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# Select the model with the highest accuracy
best_result = max(all_results, key=lambda item: item["accuracy"])
best_name = best_result["model"]
best_model = trained_models[best_name]

print("\nBest model:", best_name)
print(f"Accuracy: {best_result['accuracy']:.4f}")
print("\nResults saved to:", OUTPUT_DIR.resolve())

# 8. Save the best trained model
joblib.dump(
    best_model,
    MODEL_DIR / "sentiment_model.joblib"
)

print("Best model saved successfully.")
print("Model location:", (MODEL_DIR / "sentiment_model.joblib").resolve())

# 9. Predict sentiment for new reviews
print("\nTest the model with your own English movie reviews.")
print("Type 'quit' to exit.")

sentiment_label = {
    "pos": "Positive",
    "neg": "Negative"
}

while True:
    try:
        review = input("\nEnter a movie review: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nProgram stopped.")
        break

    if review.lower() == "quit":
        print("Goodbye!")
        break

    if not review:
        print("Please enter some text.")
        continue

    prediction = best_model.predict([review])[0]

    probabilities = best_model.predict_proba([review])[0]
    classes = list(best_model.classes_)
    confidence = probabilities[classes.index(prediction)]

    print("Predicted sentiment:", sentiment_label[prediction])
    print(f"Model probability for this prediction: {confidence:.2%}")