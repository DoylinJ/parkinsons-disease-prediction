import pandas as pd
import numpy as np
import sys
import os
import time

# Add all_features directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'all_features'))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


# Load Dataset
try:
    df = pd.read_csv("all_features/turning_pd_features.csv")
except FileNotFoundError:
    print("Dataset not found. Please ensure 'all_features/turning_pd_features.csv' exists.")
    exit(1)

X = df.drop(["label", "filename"], axis=1, errors="ignore")
y = df["label"]

print(f"\nLoaded dataset with {len(X)} samples and {X.shape[1]} features.")

# Models

models = {
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
}

# Metric Storage

def init_metrics():
    metrics = {}
    for name in models:
        metrics[name] = {
            "acc": [], "precision": [], "recall": [], "f1": [],
            "error": [], "time": []
        }
    return metrics

metrics_original = init_metrics()
metrics_pca = init_metrics()
metrics_svd = init_metrics()

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-Fold Cross Validation...\n")

for train_idx, test_idx in cv.split(X, y):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # SVD
    svd = TruncatedSVD(n_components=4, random_state=42)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)

    for name, model in models.items():

        # ORIGINAL FEATURES
        start = time.time()

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        pred_time = time.time() - start

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        err = 1 - acc

        metrics_original[name]["acc"].append(acc)
        metrics_original[name]["precision"].append(prec)
        metrics_original[name]["recall"].append(rec)
        metrics_original[name]["f1"].append(f1)
        metrics_original[name]["error"].append(err)
        metrics_original[name]["time"].append(pred_time)

        # PCA FEATURES
        start = time.time()

        model.fit(X_train_pca, y_train)
        preds_pca = model.predict(X_test_pca)

        pred_time = time.time() - start

        acc = accuracy_score(y_test, preds_pca)
        prec = precision_score(y_test, preds_pca)
        rec = recall_score(y_test, preds_pca)
        f1 = f1_score(y_test, preds_pca)
        err = 1 - acc

        metrics_pca[name]["acc"].append(acc)
        metrics_pca[name]["precision"].append(prec)
        metrics_pca[name]["recall"].append(rec)
        metrics_pca[name]["f1"].append(f1)
        metrics_pca[name]["error"].append(err)
        metrics_pca[name]["time"].append(pred_time)

        # SVD FEATURES
        start = time.time()

        model.fit(X_train_svd, y_train)
        preds_svd = model.predict(X_test_svd)

        pred_time = time.time() - start

        acc = accuracy_score(y_test, preds_svd)
        prec = precision_score(y_test, preds_svd)
        rec = recall_score(y_test, preds_svd)
        f1 = f1_score(y_test, preds_svd)
        err = 1 - acc

        metrics_svd[name]["acc"].append(acc)
        metrics_svd[name]["precision"].append(prec)
        metrics_svd[name]["recall"].append(rec)
        metrics_svd[name]["f1"].append(f1)
        metrics_svd[name]["error"].append(err)
        metrics_svd[name]["time"].append(pred_time)

# Print Results
def print_results(title, metrics):

    print("\n" + "="*65)
    print(title)
    print("="*65)

    for model in metrics:

        print(f"\n{model}")
        print(f"Accuracy      : {np.mean(metrics[model]['acc']):.4f}")
        print(f"Precision     : {np.mean(metrics[model]['precision']):.4f}")
        print(f"Recall        : {np.mean(metrics[model]['recall']):.4f}")
        print(f"F1 Score      : {np.mean(metrics[model]['f1']):.4f}")
        print(f"Error Rate    : {np.mean(metrics[model]['error']):.4f}")
        print(f"Prediction Time (s): {np.mean(metrics[model]['time']):.4f}")
        print("-"*50)

# Function to Find Best Model

def get_best_model(metrics_original, metrics_pca, metrics_svd):

    best_model = None
    best_feature = None
    best_f1 = 0
    best_acc = 0

    feature_sets = {
        "Original": metrics_original,
        "PCA": metrics_pca,
        "SVD": metrics_svd
    }

    for feature_name, metrics in feature_sets.items():

        for model in metrics:

            avg_f1 = np.mean(metrics[model]["f1"])
            avg_acc = np.mean(metrics[model]["acc"])

            if avg_f1 > best_f1 or (avg_f1 == best_f1 and avg_acc > best_acc):
                best_f1 = avg_f1
                best_acc = avg_acc
                best_model = model
                best_feature = feature_name

    return best_model, best_feature, best_acc, best_f1

# Display Results

print_results("MODEL PERFORMANCE (ORIGINAL FEATURES)", metrics_original)
print_results("MODEL PERFORMANCE AFTER PCA", metrics_pca)
print_results("MODEL PERFORMANCE AFTER SVD", metrics_svd)

# Find Best Model

best_model, best_feature, best_acc, best_f1 = get_best_model(
    metrics_original,
    metrics_pca,
    metrics_svd
)

print("\n" + "="*60)
print("BEST MODEL SELECTED")
print("="*60)

print(f"Model Type     : {best_model}")
print(f"Feature Method : {best_feature}")
print(f"Accuracy       : {best_acc:.4f}")
print(f"F1 Score       : {best_f1:.4f}")