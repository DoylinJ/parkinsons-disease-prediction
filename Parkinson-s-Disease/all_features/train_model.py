import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from custom_bagging import CustomBaggingClassifier
import joblib

df = pd.read_csv("turning_pd_features.csv")
X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"Training on {len(X)} samples | {y.value_counts().to_dict()}")

base_tree = DecisionTreeClassifier(
    min_samples_leaf=4,
    max_depth=8,
    random_state=42
)

model = CustomBaggingClassifier(
    base_estimator=base_tree,
    n_estimators=1000,
    max_samples=0.75,
    max_features=0.75,
    random_state=42
)

model.fit(X, y)
joblib.dump(model, "model.pkl")
print("Model saved: model.pkl")
