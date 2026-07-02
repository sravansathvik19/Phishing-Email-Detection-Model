"""
phishing_detector.py
================================================================
Phishing Email Detection Model using Scikit-learn
================================================================

Pipeline:
1. Load email dataset (text + label)
2. Extract hand-crafted features (URLs, suspicious keywords, punctuation, etc.)
3. Combine with TF-IDF text features
4. Train a classifier (Logistic Regression / Random Forest)
5. Evaluate: accuracy, confusion matrix, classification report
6. Save trained model + vectorizer for reuse
7. Provide a predict_email() function for new/unseen emails
"""

import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Feature engineering helpers
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "click here", "confirm", "password",
    "account", "update", "security", "limited time", "act now", "winner",
    "congratulations", "free", "claim", "credit card", "bank", "ssn",
    "social security", "login", "expire", "locked", "unauthorized",
    "immediately", "alert", "refund", "gift card", "prize", "billing",
]

URL_PATTERN = re.compile(r"(https?://[^\s]+|www\.[^\s]+)")
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
SHORTENER_DOMAINS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd"]


def extract_handcrafted_features(text: str) -> dict:
    """Extract numeric/boolean features from a raw email text string."""
    text_lower = text.lower()
    urls = URL_PATTERN.findall(text)

    num_urls = len(urls)
    has_ip_url = int(bool(IP_URL_PATTERN.search(text)))
    has_shortener = int(any(domain in text_lower for domain in SHORTENER_DOMAINS))

    num_suspicious_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text_lower)

    num_exclamations = text.count("!")
    num_dollar_signs = text.count("$")
    num_uppercase_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    text_length = len(text)
    num_words = len(text.split())

    # Ratio of punctuation characters - phishing emails often have more!!! or $$$
    num_punct = sum(1 for c in text if c in string.punctuation)
    punct_ratio = num_punct / max(text_length, 1)

    return {
        "num_urls": num_urls,
        "has_ip_url": has_ip_url,
        "has_shortener": has_shortener,
        "num_suspicious_keywords": num_suspicious_keywords,
        "num_exclamations": num_exclamations,
        "num_dollar_signs": num_dollar_signs,
        "num_uppercase_words": num_uppercase_words,
        "text_length": text_length,
        "num_words": num_words,
        "punct_ratio": punct_ratio,
    }


def build_feature_dataframe(texts: pd.Series) -> pd.DataFrame:
    feats = texts.apply(extract_handcrafted_features)
    return pd.DataFrame(list(feats))


# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------

def load_data(path="emails.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df


# ---------------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    df = load_data("emails.csv")
    print(f"Total emails: {len(df)}")
    print(df["label"].value_counts(), "\n")

    X_text = df["text"]
    y = df["label"].map({"legitimate": 0, "phishing": 1})

    # Hand-crafted numeric features
    print("Extracting hand-crafted features (URLs, keywords, punctuation, etc.)...")
    feat_df = build_feature_dataframe(X_text)

    # TF-IDF text features
    print("Building TF-IDF text features...")
    tfidf = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
    )
    X_tfidf = tfidf.fit_transform(X_text)

    # Combine TF-IDF (sparse) with hand-crafted features (dense -> sparse)
    X_combined = hstack([X_tfidf, csr_matrix(feat_df.values)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ------------------------------------------------------------------
    # Train two models and pick the better one
    # ------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, preds, acc)
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))

    # Select best model
    best_name = max(results, key=lambda n: results[n][2])
    best_model, best_preds, best_acc = results[best_name]
    print(f"\n>>> Best model: {best_name} (Accuracy: {best_acc:.4f})")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, best_preds)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}\nAccuracy: {best_acc:.2%}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix plot to confusion_matrix.png")

    # ------------------------------------------------------------------
    # Feature importance (if Random Forest) — show which engineered
    # features mattered, for interpretability
    # ------------------------------------------------------------------
    if best_name == "Random Forest":
        feature_names = list(tfidf.get_feature_names_out()) + list(feat_df.columns)
        importances = best_model.feature_importances_
        handcrafted_importances = importances[-len(feat_df.columns):]
        imp_df = pd.DataFrame({
            "feature": feat_df.columns,
            "importance": handcrafted_importances
        }).sort_values("importance", ascending=False)
        print("\nHand-crafted feature importances:")
        print(imp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save model artifacts for reuse
    # ------------------------------------------------------------------
    joblib.dump(best_model, "phishing_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")
    joblib.dump(list(feat_df.columns), "feature_columns.joblib")
    print("\nSaved model to phishing_model.joblib")
    print("Saved vectorizer to tfidf_vectorizer.joblib")

    return best_model, tfidf, feat_df.columns


# ---------------------------------------------------------------------------
# 4. Predict on new emails
# ---------------------------------------------------------------------------

def predict_email(text: str, model=None, tfidf=None, feature_columns=None):
    """Classify a single email as Phishing or Safe."""
    if model is None:
        model = joblib.load("phishing_model.joblib")
    if tfidf is None:
        tfidf = joblib.load("tfidf_vectorizer.joblib")
    if feature_columns is None:
        feature_columns = joblib.load("feature_columns.joblib")

    feats = extract_handcrafted_features(text)
    feat_vec = csr_matrix([[feats[c] for c in feature_columns]])
    text_vec = tfidf.transform([text])
    combined = hstack([text_vec, feat_vec])

    pred = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred]
    return label, confidence


if __name__ == "__main__":
    model, tfidf, feature_columns = main()

    print("\n" + "=" * 60)
    print("Testing on new unseen example emails")
    print("=" * 60)

    test_emails = [
        "Dear customer, your bank account has been locked. Click http://verify-bank-login.com now to restore access.",
        "Hi John, attached is the updated budget spreadsheet for Q3. Let me know if you have questions.",
        "URGENT!!! You've won a FREE iPhone! Claim now at http://bit.ly/freegift before it's gone!!",
        "Reminder: Team standup is moved to 10am tomorrow. See you there.",
    ]

    for email in test_emails:
        label, conf = predict_email(email, model, tfidf, feature_columns)
        print(f"\nEmail: {email[:80]}...")
        print(f"Prediction: {label} (confidence: {conf:.2%})")
