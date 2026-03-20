import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_features(df, horizon=1):
    data = df.copy()
    data["next_mid"] = data["mid"].shift(-horizon)
    data["target_up"] = (data["next_mid"] > data["mid"]).astype(int)

    feature_cols = [
        "imbalance",
        "ofi",
        "spread",
        "return",
        "volume",
    ]
    depth_cols = [c for c in data.columns if c.startswith("bid_depth_") or c.startswith("ask_depth_")]
    feature_cols.extend(depth_cols)

    features = data[feature_cols].iloc[:-horizon].fillna(0.0)
    target = data["target_up"].iloc[:-horizon]
    return features, target


def time_split(X, y, train_frac=0.7, val_frac=0.15):
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    return {
        "accuracy": acc,
        "auc": auc,
        "probs": probs,
    }
