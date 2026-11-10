"""
Train and compare spam classifiers, save the best one, plus metrics and plots.

Run from the project root:  python src/train.py
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

DATA_PATH = "data/spam.csv"
MODEL_PATH = "models/spam_model.joblib"
REPORT_DIR = "reports"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=["label", "text"]).drop_duplicates()
    df["label"] = df["label"].str.lower().str.strip()
    df["y"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["text"].apply(clean_text)
    return df


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} messages  (spam={df.y.sum()}, ham={(1 - df.y).sum()})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["y"], test_size=0.2, stratify=df["y"], random_state=42)

    candidates = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42,
                                                class_weight="balanced", n_jobs=-1),
    }

    results, fitted = [], {}
    for name, clf in candidates.items():
        pipe = make_pipeline(clf)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "model": name,
            "cv_f1": round(cv, 4),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
        })
        fitted[name] = pipe
        print(f"{name:<20} cv_f1={cv:.4f}  test_f1={results[-1]['f1']:.4f}")

    res_df = pd.DataFrame(results).sort_values(["f1", "cv_f1"], ascending=False)
    res_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)
    print("\n", res_df.to_string(index=False))

    best_name = res_df.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    print(f"\nBest model: {best_name}\n")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

    # --- plots ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["ham", "spam"],
                                            cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    res_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot.bar(ax=ax)
    ax.set_ylim(0.8, 1.01)
    ax.set_title("Model comparison (test set)")
    ax.set_ylabel("score")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(f"{REPORT_DIR}/model_comparison.png", dpi=150)
    plt.close(fig)

    # top spam words (works for linear models; fall back to NB log-probs)
    tfidf, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    words = tfidf.get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        weights = None
    if weights is not None:
        top = pd.Series(weights, index=words).nlargest(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        top[::-1].plot.barh(ax=ax, color="crimson")
        ax.set_title("Top words indicating SPAM")
        fig.tight_layout()
        fig.savefig(f"{REPORT_DIR}/top_spam_words.png", dpi=150)
        plt.close(fig)

    # retrain best on the full data and save
    final = make_pipeline(candidates[best_name].__class__(**candidates[best_name].get_params()))
    final.fit(df["clean"], df["y"])
    joblib.dump(final, MODEL_PATH)

    with open(f"{REPORT_DIR}/metrics.json", "w") as f:
        json.dump({"best_model": best_name,
                   "n_samples": int(len(df)),
                   "test_metrics": res_df.iloc[0].to_dict()}, f, indent=2, default=str)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
