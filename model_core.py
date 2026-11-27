# ============================================================
# model_core.py — Shared ML logic for ALL frontends
# (Streamlit app.py, Gradio app, tests — same model, same results)
# Retrains the best models at startup (deterministic, random_state=42)
# so no external model file is required -> nothing can break.
# ============================================================
import os, urllib.request, zipfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAT_PATH = os.path.join(DATA_DIR, "student-mat.csv")
POR_PATH = os.path.join(DATA_DIR, "student-por.csv")


# ---------------- Dataset ----------------
def ensure_dataset():
    """Download UCI dataset if CSVs are missing (same source as the notebook)."""
    if os.path.exists(MAT_PATH) and os.path.exists(POR_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    zip_path = os.path.join(DATA_DIR, "student-performance.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    nested = os.path.join(DATA_DIR, "student.zip")
    if os.path.exists(nested):
        with zipfile.ZipFile(nested) as z:
            z.extractall(DATA_DIR)
    if os.path.exists(zip_path):
        os.remove(zip_path)


def load_dataframe():
    """Load + merge both subjects (1044 students), create result target."""
    ensure_dataset()
    mat = pd.read_csv(MAT_PATH, sep=";")
    por = pd.read_csv(POR_PATH, sep=";")
    mat["subject"] = "math"
    por["subject"] = "portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["result"] = np.where(df["G3"] >= 10, "Pass", "Fail")
    return df


# ---------------- Training (runs once at startup) ----------------
_TRAINED = None

def get_trained():
    """Train best classifier (Logistic Regression) + best regressor
    (Random Forest) exactly like the notebook -> returns a dict."""
    global _TRAINED
    if _TRAINED is not None:
        return _TRAINED

    df = load_dataframe()
    X_raw = df.drop(columns=["G3", "result"])
    cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(X_raw, columns=cat_cols)
    y_cls = df["result"]
    y_reg = df["G3"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls)

    # --- Best classifier (same as notebook) ---
    clf = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000))])
    clf.fit(X_train, yc_train)

    # Threshold chosen by 5-fold CV on TRAIN data only (no leakage)
    oof = cross_val_predict(clf, X_train, yc_train, cv=5,
                            method="predict_proba")[:, 1]  # column 1 = 'Pass'
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        acc = accuracy_score(yc_train, np.where(oof >= t, "Pass", "Fail"))
        if acc > best_acc:
            best_t, best_acc = t, acc
    pred_test = np.where(clf.predict_proba(X_test)[:, 1] >= best_t, "Pass", "Fail")
    clf_acc = accuracy_score(yc_test, pred_test)

    # --- Best regressor (same as notebook) ---
    reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    reg.fit(X_train, yr_train)
    reg_pred = reg.predict(X_test)
    reg_r2 = r2_score(yr_test, reg_pred)
    reg_rmse = float(np.sqrt(mean_squared_error(yr_test, reg_pred)))

    _TRAINED = {
        "clf": clf, "reg": reg, "threshold": best_t, "columns": list(X.columns),
        "clf_accuracy": float(clf_acc), "reg_r2": float(reg_r2),
        "reg_rmse": reg_rmse, "n_students": len(df), "n_features": X.shape[1],
    }
    return _TRAINED


# ---------------- Prediction ----------------
def predict(student: dict):
    """student: dict with all 33 raw feature values (like the notebook demo).
    Returns dict with result, probability, score, advice."""
    t = get_trained()
    row = pd.DataFrame([student])
    enc = pd.get_dummies(row).reindex(columns=t["columns"], fill_value=0)

    proba_pass = float(t["clf"].predict_proba(enc)[0][
        list(t["clf"].classes_).index("Pass")])
    result = "Pass" if proba_pass >= t["threshold"] else "Fail"
    score = float(t["reg"].predict(enc)[0])
    score = max(0.0, min(20.0, score))          # keep inside 0-20

    if score >= 16:
        advice = "Excellent! Keep this performance up."
    elif score >= 10:
        advice = "Likely to pass. Improve G1/G2 to score above 15."
    else:
        advice = "High risk of failing. More study time, fewer absences."

    return {"result": result, "probability": proba_pass, "score": round(score, 2),
            "advice": advice, "threshold": t["threshold"]}


def default_strong_student():
    """Same strong student as the notebook demo (G1=15, G2=16)."""
    return dict(school="GP", sex="M", age=16, address="U", famsize="GT3",
                Pstatus="T", Medu=4, Fedu=3, Mjob="teacher", Fjob="services",
                reason="course", guardian="mother", studytime=3, failures=0,
                schoolsup="no", famsup="yes", paid="yes", activities="yes",
                nursery="yes", higher="yes", internet="yes", romantic="no",
                famrel=4, freetime=3, goout=2, Dalc=1, Walc=1, health=4,
                absences=2, G1=15, G2=16, subject="math")


if __name__ == "__main__":
    import time
    t0 = time.time()
    t = get_trained()
    print(f"Trained in {time.time()-t0:.1f}s")
    print(f"Classifier accuracy: {t['clf_accuracy']*100:.2f}% | "
          f"Threshold: {t['threshold']:.2f} | Features: {t['n_features']}")
    print(f"Regressor R2: {t['reg_r2']:.3f} | RMSE: {t['reg_rmse']:.2f}")
    r1 = predict(default_strong_student())
    print("Strong student ->", r1)
    weak = dict(default_strong_student(), school="MS", sex="F", age=18,
                address="R", famsize="LE3", Pstatus="A", Medu=1, Fedu=1,
                Mjob="at_home", Fjob="other", reason="reputation",
                guardian="other", studytime=1, failures=2, schoolsup="yes",
                famsup="no", paid="no", activities="no", nursery="no",
                internet="no", romantic="yes", famrel=2, freetime=4, goout=5,
                Dalc=3, Walc=4, health=3, absences=12, G1=8, G2=7)
    r2 = predict(weak)
    print("Weak student    ->", r2)
