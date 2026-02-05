# models.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import numpy as np

def run_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    models = {
        "logistic": {
            "estimator": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1.0, 10.0]}
        },
        "svc": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
            "params": {"svc__C": [0.1, 1.0], "svc__kernel": ["linear", "rbf"]}
        },
        "rf": {
            "estimator": RandomForestClassifier(),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}
        }
    }
    return _train_models(models, X_train, y_train, X_test, y_test, task="classification")

def run_regression_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = {
        "ridge": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            "params": {"ridge__alpha": [0.1, 1.0, 10.0]}
        },
        "svr": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
            "params": {"svr__C": [0.1, 1.0], "svr__kernel": ["linear", "rbf"]}
        },
        "rf": {
            "estimator": RandomForestRegressor(),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}
        }
    }
    return _train_models(models, X_train, y_train, X_test, y_test, task="regression")

def _train_models(models, X_train, y_train, X_test, y_test, task="classification"):
    results = {}
    for name, model_def in models.items():
        print(f"[Training] {name}")
        grid = GridSearchCV(model_def["estimator"], model_def["params"], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        if task == "classification":
            score = accuracy_score(y_test, y_pred)
        else:
            score = mean_squared_error(y_test, y_pred, squared=False)

        results[name] = {
            "score": score,
            "model": best_model,
            "cv_scores": cross_val_score(best_model, X_train, y_train, cv=3).tolist()
        }
    return results

def save_best_model(model, filename):
    joblib.dump(model, filename)
