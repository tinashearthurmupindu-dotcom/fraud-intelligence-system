"""
models/fraud_models.py
Trains and evaluates 5 supervised classifiers with SMOTE balancing.
Algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

FEATURE_COLS = [
    "policy_age_months", "claim_amount_usd", "claimant_age",
    "beneficiary_count", "days_to_claim", "prior_claims",
    "num_policies_same_beneficiary", "documentation_score",
    "registry_verified", "staff_relationship_flag",
    "same_day_claim", "cross_insurer_duplicate",
]

TARGET_COL = "is_fraud"


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds_roc: np.ndarray
    proba: np.ndarray
    y_true: np.ndarray
    feature_importance: Optional[Dict[str, float]] = None
    cv_auc: Optional[float] = None
    cv_std: Optional[float] = None


def prepare_data(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].values
    return X, y


def train_all_models(df: pd.DataFrame) -> Dict[str, ModelResult]:
    X, y = prepare_data(df)

    # Stratified split (70/15/15 as per proposal §3.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, stratify=y_test, random_state=42
    )

    # SMOTE on TRAINING SET ONLY (§3.3 — critical note)
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Scaler for LR and SVM
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sm)
    X_test_sc = scaler.transform(X_test)

    results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ── 1. Logistic Regression (baseline) ────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train_sc, y_train_sm)
    results["Logistic Regression"] = _evaluate(
        "Logistic Regression", lr, X_test_sc, y_test,
        feature_names=FEATURE_COLS,
        importance_attr="coef_",
        cv_estimator=LogisticRegression(max_iter=1000, C=0.5),
        cv_X=X_train_sc, cv_y=y_train_sm, cv=cv,
    )

    # ── 2. Decision Tree ──────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42)
    dt.fit(X_train_sm, y_train_sm)
    results["Decision Tree"] = _evaluate(
        "Decision Tree", dt, X_test, y_test,
        feature_names=FEATURE_COLS,
        importance_attr="feature_importances_",
        cv_estimator=DecisionTreeClassifier(max_depth=8),
        cv_X=X_train_sm, cv_y=y_train_sm, cv=cv,
    )

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_train_sm, y_train_sm)
    results["Random Forest"] = _evaluate(
        "Random Forest", rf, X_test, y_test,
        feature_names=FEATURE_COLS,
        importance_attr="feature_importances_",
        cv_estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
        cv_X=X_train_sm, cv_y=y_train_sm, cv=cv,
    )

    # ── 4. XGBoost ────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        scale_pos = int((y_train_sm == 0).sum() / max((y_train_sm == 1).sum(), 1))
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        xgb.fit(X_train_sm, y_train_sm)
        results["XGBoost"] = _evaluate(
            "XGBoost", xgb, X_test, y_test,
            feature_names=FEATURE_COLS,
            importance_attr="feature_importances_",
            cv_estimator=XGBClassifier(n_estimators=100, verbosity=0),
            cv_X=X_train_sm, cv_y=y_train_sm, cv=cv,
        )
    else:
        # Fallback extra RF if XGBoost not installed
        results["XGBoost*"] = results.get("Random Forest")

    # ── 5. SVM ────────────────────────────────────────────────────────────────
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_sc, y_train_sm)
    results["SVM"] = _evaluate(
        "SVM", svm, X_test_sc, y_test,
        feature_names=FEATURE_COLS,
        importance_attr=None,
        cv_estimator=SVC(kernel="rbf", probability=True),
        cv_X=X_train_sc, cv_y=y_train_sm, cv=cv,
    )

    return results


def _evaluate(
    name, model, X_test, y_test,
    feature_names, importance_attr,
    cv_estimator, cv_X, cv_y, cv,
) -> ModelResult:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    fpr, tpr, thr = roc_curve(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    # Feature importance
    fi = None
    if importance_attr and hasattr(model, importance_attr):
        raw = getattr(model, importance_attr)
        if raw.ndim > 1:
            raw = np.abs(raw).mean(axis=0)
        fi = dict(zip(feature_names, raw.tolist()))

    # Cross-val AUC
    cv_scores = cross_val_score(
        cv_estimator, cv_X, cv_y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    return ModelResult(
        name=name,
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        auc_roc=roc_auc_score(y_test, proba),
        cm=cm,
        fpr=fpr, tpr=tpr, thresholds_roc=thr,
        proba=proba, y_true=y_test,
        feature_importance=fi,
        cv_auc=cv_scores.mean(),
        cv_std=cv_scores.std(),
    )


def score_single_claim(claim: dict, model, scaler=None) -> dict:
    """
    Score a single claim dict and return fraud probability + risk tier.
    """
    X = pd.DataFrame([{k: claim.get(k, 0) for k in FEATURE_COLS}])
    if scaler:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[0, 1]
    if proba >= 0.75:
        tier = "HIGH"
    elif proba >= 0.45:
        tier = "MEDIUM"
    else:
        tier = "LOW"
    return {"fraud_probability": round(float(proba), 4), "risk_tier": tier}


def results_to_dataframe(results: Dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(r.accuracy, 4),
                "Precision": round(r.precision, 4),
                "Recall": round(r.recall, 4),
                "F1-Score": round(r.f1, 4),
                "AUC-ROC": round(r.auc_roc, 4),
                "CV AUC (mean)": round(r.cv_auc, 4) if r.cv_auc else "—",
                "CV Std": round(r.cv_std, 4) if r.cv_std else "—",
            }
        )
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
