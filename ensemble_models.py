from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from grader_contracts.ensembles import EnsembleInput, EnsembleResult

NUMERIC_COLUMNS = ["Duration_of_Credit_monthly", "Credit_Amount", "Age_years"]


def make_model(kind: str, seed: int = 42) -> Pipeline:
    """Одна предобработка для всех сравниваемых ансамблей."""
    if kind == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)
    elif kind == "Extra Trees":
        classifier = ExtraTreesClassifier(n_estimators=250, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1)
    elif kind == "Gradient Boosting":
        classifier = GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=seed)
    elif kind == "Random Forest (leaf=1)":
        classifier = RandomForestClassifier(n_estimators=250, min_samples_leaf=1, max_features="sqrt", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Неизвестная модель: {kind}")
    return Pipeline([("preprocessing", "passthrough"), ("classifier", classifier)])


def prepare_model(kind: str, features: pd.DataFrame, seed: int = 42) -> Pipeline:
    numeric = [col for col in NUMERIC_COLUMNS if col in features.columns]
    categorical = [col for col in features.columns if col not in numeric]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = make_model(kind, seed)
    model.set_params(preprocessing=preprocess)
    return model


def to_frame(features, names=None) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()
    return pd.DataFrame(features, columns=names)


def calculate_metrics(target, probabilities) -> dict:
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает выбранный ансамбль на train и считает метрики на test."""
    train = to_frame(data.train_features)
    test = to_frame(data.test_features, train.columns)
    model_name = "Random Forest"
    model = prepare_model(model_name, train, data.random_seed)
    model.fit(train, np.asarray(data.train_target).ravel())
    probabilities = model.predict_proba(test)[:, list(model.named_steps["classifier"].classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    scores = calculate_metrics(data.test_target, probabilities)
    return EnsembleResult(model_name=model_name, probabilities=probabilities, predictions=predictions, **scores)
