# Phishing Email Detection Model

## Files
- `generate_dataset.py` — generates `emails.csv` (synthetic phishing + legitimate emails). Swap in a real dataset (e.g. Kaggle phishing email datasets) by replacing this file's output.
- `phishing_detector.py` — main pipeline: feature extraction, TF-IDF, model training, evaluation, saving, and prediction on new emails.
- `emails.csv` — the generated dataset (800 emails, 50/50 split).
- `phishing_model.joblib`, `tfidf_vectorizer.joblib`, `feature_columns.joblib` — saved trained artifacts.
- `confusion_matrix.png` — confusion matrix plot from the last run.

## How to run
```bash
pip install scikit-learn pandas matplotlib seaborn joblib
python3 generate_dataset.py      # creates emails.csv
python3 phishing_detector.py     # trains, evaluates, saves model, tests sample emails
```

## What it does
1. **Feature extraction**: number of URLs, IP-based URLs, link shorteners, suspicious keywords (urgent, verify, password, etc.), exclamation marks, dollar signs, uppercase words, punctuation ratio, text length.
2. **Text features**: TF-IDF (1-2 grams, top 2000 terms) combined with the hand-crafted features above.
3. **Models**: trains both Logistic Regression and Random Forest, picks the better one on accuracy.
4. **Evaluation**: prints accuracy, precision/recall/F1, confusion matrix, and saves a confusion matrix plot.
5. **Inference**: `predict_email(text)` classifies any new email as "Phishing" or "Safe" with a confidence score.

## Using your own / a real dataset
Replace `emails.csv` with a real one — it just needs two columns: `text` and `label` (`phishing` / `legitimate`). Then re-run `phishing_detector.py`. No other code changes needed.

## Note on accuracy
The included synthetic dataset is template-based, so the model scores ~100% — that reflects the cleanliness of the synthetic data, not real-world performance. On a real, noisy dataset (e.g. Nazario phishing corpus + Enron legitimate emails) expect ~90-97% accuracy, which is still very strong for this kind of text classification task.
