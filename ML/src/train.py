"""
src/train.py
------------
Split → fit → evaluate → save.

Functions
---------
train(X, y)              — stratified 80/20 split, fit XGBClassifier, return model
evaluate(model, X, y)    — print accuracy + classification report
save_model(model, path)  — persist model with joblib
load_model(path)         — reload model from disk
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DEFAULT_MODEL_PATH = os.path.join("models", "blunder_model.pkl")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train/test split, then fit an XGBClassifier.

    The class imbalance is handled automatically via scale_pos_weight:
        scale_pos_weight = count(non-blunders) / count(blunders)

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (output of build_dataset).
    y : pd.Series
        Binary blunder labels.
    test_size : float
        Fraction of data held out for evaluation (default 0.20).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    model : XGBClassifier
        Trained model.
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Class-imbalance correction
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={neg:,} / pos={pos:,})")

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=600,
        max_depth=8,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    print("Training XGBoost …")
    model.fit(X_train, y_train)
    print("Training complete.")

    return model, X_test, y_test


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Print accuracy and a full classification report.

    Parameters
    ----------
    model : XGBClassifier
        Trained model.
    X_test : pd.DataFrame
    y_test : pd.Series
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred) * 100

    print("\n" + "=" * 30)
    print("XGBOOST RESULTS")
    print("=" * 30)
    print(f"Accuracy: {acc:.2f} %\n")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Blunder"]))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: XGBClassifier, path: str = DEFAULT_MODEL_PATH) -> None:
    """
    Save the trained model to disk using joblib.

    Parameters
    ----------
    model : XGBClassifier
    path : str
        Destination file path (default: models/blunder_model.pkl).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model(path: str = DEFAULT_MODEL_PATH) -> XGBClassifier:
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : str
        Path to the .pkl file (default: models/blunder_model.pkl).

    Returns
    -------
    XGBClassifier
    """
    model = joblib.load(path)
    print(f"Model loaded ← {path}")
    return model
