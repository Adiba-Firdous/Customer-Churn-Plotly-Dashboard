"""
Optional adapter example for a real customer churn dataset.
Expected input: a CSV containing the target column `churn_flag`.
This script trains a baseline model and writes held-out predictions for app.py.
It does NOT claim to reproduce a previous model; replace the estimator/preprocessing
with the exact model pipeline from your earlier task when available.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
input_path = ROOT / "data" / "customer_churn_analysis_ready.csv"
output_path = ROOT / "outputs" / "model_predictions.csv"
TARGET = "churn_flag"

if not input_path.exists():
    raise FileNotFoundError(
        f"Place your prepared dataset at {input_path}. It must contain `{TARGET}`."
    )

df = pd.read_csv(input_path)
if TARGET not in df.columns:
    raise ValueError(f"Target column `{TARGET}` was not found.")
df = df.dropna(subset=[TARGET]).copy()
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])
# Exclude identifiers to prevent the model learning customer IDs.
X = X.drop(columns=[c for c in X.columns if c.lower() in {"customer_id", "id"}], errors="ignore")

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
out = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": pred, "y_probability": prob})
output_path.parent.mkdir(exist_ok=True)
out.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall   :", recall_score(y_test, pred, zero_division=0))
print("F1-score :", f1_score(y_test, pred, zero_division=0))
