"""
Predict whether a message is spam.

Usage:
    python src/predict.py "You have won $1000, click http://bit.ly/x"
    python src/predict.py            # interactive mode
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_model.joblib")
_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_message(text: str) -> dict:
    model = _load()
    clean = clean_text(text)
    label = int(model.predict([clean])[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([clean])[0][1])
    else:  # e.g. LinearSVC: squash decision score into 0..1
        score = float(model.decision_function([clean])[0])
        prob = 1 / (1 + 2.718281828 ** (-score))
    return {"label": "spam" if label else "ham", "spam_probability": round(prob, 4)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        print(predict_message(msg))
    else:
        print("Type a message (empty line to quit)")
        while True:
            msg = input("> ").strip()
            if not msg:
                break
            print(predict_message(msg))
