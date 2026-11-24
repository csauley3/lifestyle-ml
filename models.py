import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# This project uses CDC data to see if we can predict who had a heart attack
# based on their lifestyle habits/health.

# load cleaned dataset select ifestyle/health columns for heart attack
df = pd.read_csv("heart_2022_no_nans.csv")

# These columns were chosen. More exist in the dataset but are somewhat irrelavant.
feature_cols = [
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "LastCheckupTime",
    "PhysicalActivities",
    "SleepHours",
    "SmokerStatus",
    "ECigaretteUsage",
    "AgeCategory",
    "BMI",
    "AlcoholDrinkers",
]

X = df[feature_cols]
y = (df["HadHeartAttack"] == "Yes").astype(int)

# split 80/20
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# preprocessing: scale the numeric columns & one-hot encode the text columns
num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
)

# model 1: Logistic Regression
log_model = make_pipeline(
    preprocessor,
    LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42),
)
log_model.fit(X_tr, y_tr)

# model 2: Linear SVM
svm_model = make_pipeline(
    preprocessor,
    LinearSVC(C=10.0, class_weight="balanced", dual=False, random_state=42),
)
svm_model.fit(X_tr, y_tr)

# train vs. test performance
def get_metrics(name, split, y_true, y_pred, y_score):
    return {
        "Model": name,
        "Split": split,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

log_tr_sc = log_model.predict_proba(X_tr)[:, 1]
log_te_sc = log_model.predict_proba(X_te)[:, 1]
svm_tr_sc = svm_model.decision_function(X_tr)
svm_te_sc = svm_model.decision_function(X_te)

results_df = pd.DataFrame([
    get_metrics("Logistic Reg (C=0.1)", "Train", y_tr, log_model.predict(X_tr), log_tr_sc),
    get_metrics("Logistic Reg (C=0.1)", "Test",  y_te, log_model.predict(X_te), log_te_sc),
    get_metrics("Linear SVM (C=10.0)",  "Train", y_tr, svm_model.predict(X_tr), svm_tr_sc),
    get_metrics("Linear SVM (C=10.0)",  "Test",  y_te, svm_model.predict(X_te), svm_te_sc),
])
print("\n", results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

cm_log = confusion_matrix(y_te, log_model.predict(X_te))
cm_svm = confusion_matrix(y_te, svm_model.predict(X_te))
print("\nTest Confusion Matrix — Logistic Regression:\n", cm_log)
print("\nTest Confusion Matrix — Linear SVM:\n", cm_svm)

# plot ROC curves and cms
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

fpr_l, tpr_l, _ = roc_curve(y_te, log_te_sc)
fpr_s, tpr_s, _ = roc_curve(y_te, svm_te_sc)
axes[0].plot(fpr_l, tpr_l, color="#1f77b4", lw=3, label=f"Logistic Reg (AUC = {roc_auc_score(y_te, log_te_sc):.4f})")
axes[0].plot(fpr_s, tpr_s, color="#ff7f0e", lw=2, linestyle="--", label=f"Linear SVM (AUC = {roc_auc_score(y_te, svm_te_sc):.4f})")
axes[0].plot([0, 1], [0, 1], "k:", label="Random Baseline")
axes[0].set_title("ROC Curves (11 Lifestyle Features)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right", fontsize=8.5)
axes[0].grid(alpha=0.3)

sns.heatmap(cm_log, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[1],
            xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"])
axes[1].set_title("Logistic Reg (C=0.1) Confusion Matrix", fontweight="bold")

sns.heatmap(cm_svm, annot=True, fmt=",d", cmap="Oranges", cbar=False, ax=axes[2],
            xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"])
axes[2].set_title("Linear SVM (C=10.0) Confusion Matrix", fontweight="bold")

plt.tight_layout()
plt.savefig("lifestyle_11cols_comparison.png", dpi=150, bbox_inches="tight")
plt.show()