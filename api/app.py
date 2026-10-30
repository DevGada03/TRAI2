"""
PrivateMed FastAPI Backend
Delivering endpoints for Module 4 requirements:
- Laplace Mechanism & DP-SGD Clinical Training
- Sensitivity & Feature Privacy Allocation
- Pareto Analysis (AUC-ROC vs. Epsilon)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import math

try:
    from dp_engine import (
        DPClinicalClassifier,
        generate_clinical_cohort,
        evaluate_metrics,
        generate_pareto_curve,
        calculate_l1_sensitivity,
        laplace_mechanism
    )
except ImportError:
    from api.dp_engine import (
        DPClinicalClassifier,
        generate_clinical_cohort,
        evaluate_metrics,
        generate_pareto_curve,
        calculate_l1_sensitivity,
        laplace_mechanism
    )

app = FastAPI(
    title="PrivateMed: Differentially Private Clinical Model Training",
    description="Module 4: Privacy Vulnerabilities, Laplace Mechanism, (ε, δ)-Differential Privacy",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    epsilon: float = 2.0
    mechanism: str = "laplace"  # "laplace" or "dp_sgd"
    clip_norm: float = 1.0
    epochs: int = 25
    delta: float = 1e-5


class LaplaceQueryRequest(BaseModel):
    feature_name: str
    true_mean: float
    sensitivity: float
    epsilon: float


@app.get("/health")
def health():
    return {"status": "active", "module": "Module 4 Differential Privacy"}


@app.post("/train")
def train_model(req: TrainRequest):
    X, y, feature_names = generate_clinical_cohort(n_patients=1200, seed=42)
    split = 800
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    mode = "laplace_feature" if req.mechanism == "laplace" else "dp_sgd"
    clf = DPClinicalClassifier(epsilon=req.epsilon, clip_norm=req.clip_norm, mode=mode)
    clf.fit(X_train, y_train, epochs=req.epochs, lr=0.08, seed=42)

    y_prob = clf.predict_proba(X_test)
    acc, auc = evaluate_metrics(y_test, y_prob)

    # Theoretical Membership Inference Attack (MIA) bound under ε-DP
    exp_eps = math.exp(min(req.epsilon, 20.0))
    mia_bound = (exp_eps - 1.0) / (exp_eps + 1.0)
    snoop_accuracy = round(0.50 + 0.50 * mia_bound * 0.65, 4)

    # Privacy budget allocation per feature (5 clinical features)
    d = len(feature_names)
    eps_per_feature = round(req.epsilon / d, 4)

    return {
        "epsilon": req.epsilon,
        "delta": req.delta if req.mechanism == "dp_sgd" else 0.0,
        "mechanism": req.mechanism,
        "accuracy": acc,
        "auc_roc": auc,
        "snoop_accuracy": min(0.92, snoop_accuracy),
        "feature_budget_allocation": {
            feat: eps_per_feature for feat in feature_names
        },
        "l1_global_sensitivity": 5.0,
        "laplace_scale_b": round(5.0 / max(0.01, req.epsilon), 4)
    }


@app.post("/query_laplace")
def query_laplace(req: LaplaceQueryRequest):
    noisy_val = laplace_mechanism(
        true_value=req.true_mean,
        sensitivity=req.sensitivity,
        epsilon=req.epsilon
    )
    scale = req.sensitivity / req.epsilon
    return {
        "feature": req.feature_name,
        "true_value": req.true_mean,
        "noisy_value": round(noisy_val, 2),
        "scale_parameter_b": round(scale, 4),
        "noise_deviation": round(noisy_val - req.true_mean, 2)
    }


@app.get("/pareto_curve")
def get_pareto_curve():
    data = generate_pareto_curve()
    return {"pareto_frontier": data}
