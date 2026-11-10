# Spam Detection using Machine Learning (Python)

Classifies SMS/email messages as **spam** or **ham** (legitimate) using NLP (TF-IDF) and
classical ML models. Includes training, evaluation, a command-line predictor and a Flask web app.

## Project structure

```
spam_detection/
├── app.py                    # Flask web app
├── requirements.txt
├── README.md
├── data/
│   ├── spam.csv              # SMS Spam Collection (5,574 msgs: 4,827 ham / 747 spam)
│   └── synthetic_spam.csv    # small synthetic dataset (optional, from generate_dataset.py)
├── src/
│   ├── preprocess.py         # text cleaning
│   ├── train.py              # trains 4 models, picks best, saves plots/metrics
│   ├── predict.py            # CLI / function for predictions
│   └── generate_dataset.py   # synthetic dataset generator
├── models/
│   └── spam_model.joblib     # trained pipeline (TF-IDF + best classifier)
├── reports/
│   ├── model_comparison.csv / .png
│   ├── confusion_matrix.png
│   ├── top_spam_words.png
│   └── metrics.json
└── templates/
    └── index.html            # web UI
```

## How to run

```bash
pip install -r requirements.txt
python src/train.py                      # train + evaluate (creates models/ and reports/)
python src/predict.py "You won $1000, call 0800123456 now"   # CLI prediction
python app.py                            # web app at http://127.0.0.1:5000
```

## Method

1. **Data**: SMS Spam Collection (UCI). Duplicates removed -> 5,171 unique messages.
2. **Preprocessing**: lowercase; URLs, phone numbers and digits replaced by tokens
   (`urltoken`, `phonetoken`, `numtoken`); punctuation removed.
3. **Features**: TF-IDF with unigrams + bigrams (sublinear TF).
4. **Models compared**: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest.
   80/20 stratified split + 5-fold cross-validation on the training set.
5. **Selection**: best test F1 on the spam class. The final model is retrained on all data.

## Results (test set, 1,035 messages)

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| **Linear SVM**      | 0.9894   | 0.9839    | 0.9313 | 0.9569 |
| Naive Bayes         | 0.9865   | 0.9756    | 0.9160 | 0.9449 |
| Logistic Regression | 0.9836   | 0.9254    | 0.9466 | 0.9358 |
| Random Forest       | 0.9845   | 1.0000    | 0.8779 | 0.9350 |

Because the data is imbalanced (~13% spam), F1/recall on the spam class matter more than
accuracy alone.

## Notes / limitations

- The "spam probability" shown for Linear SVM is a sigmoid of the decision score, so treat it
  as a rough confidence, not a calibrated probability.
- The dataset is from 2000s-era SMS; modern phishing/spam may differ. Retrain with newer
  data (any CSV with `label` = ham/spam and `text` columns) to improve it.

## Future improvements

Word-embedding / BERT models, probability calibration, deploying the app online,
and a feedback loop that retrains on user-flagged messages.
