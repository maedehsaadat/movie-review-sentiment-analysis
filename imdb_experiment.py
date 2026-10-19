
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Set paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "results_imdb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the IMDb dataset
print("Loading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

train_df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "sentiment": dataset["train"]["label"]
})

test_df = pd.DataFrame({
    "text": dataset["test"]["text"],
    "sentiment": dataset["test"]["label"]
})

# Convert numeric labels to the same format as the original project
label_mapping = {
    0: "neg",
    1: "pos"
}

train_df["sentiment"] = train_df["sentiment"].map(label_mapping)
test_df["sentiment"] = test_df["sentiment"].map(label_mapping)

print("Training dataset size:", len(train_df))
print("Test dataset size:", len(test_df))

print("\nTraining class distribution:")
print(train_df["sentiment"].value_counts())

print("\nTest class distribution:")
print(test_df["sentiment"].value_counts())

train_df["sentiment"].value_counts().rename_axis(
    "sentiment"
).reset_index(name="count").to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

# 3. Use IMDb's official train/test split
X_train = train_df["text"]
y_train = train_df["sentiment"]

X_test = test_df["text"]
y_test = test_df["sentiment"]

# 4. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

all_results = []
trained_models = {}

# 5. Train and evaluate models
for name, classifier in models.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            dtype="float32"
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

    filename = name.lower().replace(" ", "_")

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(
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

    plt.title(f"IMDb Confusion Matrix - {name}")
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

# 6. Compare models
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_DIR / "imdb_model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("IMDb MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 7. Select the model with the highest accuracy
best_result = max(all_results, key=lambda item: item["accuracy"])
best_name = best_result["model"]
best_model = trained_models[best_name]

print("\nBest IMDb model:", best_name)
print(f"Accuracy: {best_result['accuracy']:.4f}")

# Save under a NEW filename so the original model remains unchanged
best_model_path = MODEL_DIR / "imdb_sentiment_model.joblib"

joblib.dump(best_model, best_model_path)

print("\nIMDb results saved to:", OUTPUT_DIR.resolve())
print("IMDb model saved to:", best_model_path.resolve())

# 8. Interactive prediction using the best IMDb model
print("\nTest the IMDb model with your own English movie reviews.")
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
