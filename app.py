"""Simple Flask web app for the spam detector.  Run: python app.py  ->  http://127.0.0.1:5000"""
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_message  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a message."}), 400
    return jsonify(predict_message(text))


if __name__ == "__main__":
    app.run(debug=True)
