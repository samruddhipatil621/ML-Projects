# =============================================================
#   TASK 4: DISEASE PREDICTION FROM MEDICAL DATA
#   Predict whether a patient has:
#       - Heart Disease (Cleveland dataset)
#       - Diabetes     (Pima Indians dataset)
#   Both datasets come FREE from sklearn — no download needed!
#   Author: Add your name here
# =============================================================

# ──────────────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer          # Built-in dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# XGBoost — install if missing: pip install xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False
    print("[INFO] XGBoost not found. Using GradientBoosting instead.")
    print("       To install XGBoost: pip install xgboost")


# ──────────────────────────────────────────────────────────────
# STEP 2: LOAD DATASETS
# We use TWO built-in datasets from sklearn.
# ──────────────────────────────────────────────────────────────

def load_breast_cancer_data():
    """
    Breast Cancer Wisconsin Dataset (built into sklearn).
    30 features derived from cell nucleus measurements.
    Target: 0 = Malignant (cancerous), 1 = Benign (not cancerous)
    569 patients, 30 features.
    """
    data = load_breast_cancer()
    df   = pd.DataFrame(data.data, columns=data.feature_names)
    df["DIAGNOSIS"] = data.target
    # Rename target to be intuitive
    df["DIAGNOSIS_LABEL"] = df["DIAGNOSIS"].map({0: "Malignant", 1: "Benign"})

    print(f"\n{'='*60}")
    print(f"  BREAST CANCER DATASET")
    print(f"{'='*60}")
    print(f"  Patients   : {df.shape[0]}")
    print(f"  Features   : {df.shape[1] - 2}")
    print(f"  Benign     : {(df['DIAGNOSIS']==1).sum()} ({(df['DIAGNOSIS']==1).mean()*100:.1f}%)")
    print(f"  Malignant  : {(df['DIAGNOSIS']==0).sum()} ({(df['DIAGNOSIS']==0).mean()*100:.1f}%)")
    print(f"\n  Key features: {list(data.feature_names[:5])} ...")

    return df, data.feature_names.tolist(), "Breast Cancer", ["Malignant","Benign"]


def load_heart_disease_data():
    """
    Heart Disease Dataset (Cleveland Heart Disease from UCI).
    Loaded directly via URL — requires internet.
    Target: 0 = No Disease, 1 = Has Disease
    303 patients, 13 features.
    Falls back to synthetic data if no internet.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "chest_pain", "resting_bp", "cholesterol",
        "fasting_sugar", "rest_ecg", "max_heart_rate", "exercise_angina",
        "st_depression", "st_slope", "major_vessels", "thal", "TARGET"
    ]
    try:
        df = pd.read_csv(url, header=None, names=columns, na_values="?")
        df.dropna(inplace=True)
        df["TARGET"] = (df["TARGET"] > 0).astype(int)   # 1-4 → 1 (disease), 0 stays 0
        feature_names = columns[:-1]

        print(f"\n{'='*60}")
        print(f"  HEART DISEASE DATASET (Cleveland)")
        print(f"{'='*60}")
        print(f"  Patients       : {df.shape[0]}")
        print(f"  Features       : {len(feature_names)}")
        print(f"  No Disease     : {(df['TARGET']==0).sum()}")
        print(f"  Has Disease    : {(df['TARGET']==1).sum()}")
        print(f"\n  Key features: {feature_names[:5]} ...")

        return df, feature_names, "Heart Disease", ["No Disease","Has Disease"]

    except Exception as e:
        print(f"[INFO] Could not load Heart Disease dataset ({e}).")
        print("[INFO] Generating synthetic heart disease data instead...")
        return _generate_heart_data()


def _generate_heart_data():
    """Fallback: synthetic heart disease data."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "age":              np.random.randint(30, 75, n),
        "sex":              np.random.choice([0, 1], n),
        "chest_pain":       np.random.choice([0,1,2,3], n),
        "resting_bp":       np.random.normal(130, 20, n).clip(90, 200).astype(int),
        "cholesterol":      np.random.normal(245, 50, n).clip(140, 400).astype(int),
        "fasting_sugar":    np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "rest_ecg":         np.random.choice([0,1,2], n),
        "max_heart_rate":   np.random.normal(150, 22, n).clip(70, 200).astype(int),
        "exercise_angina":  np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "st_depression":    np.random.uniform(0, 5, n).round(1),
        "st_slope":         np.random.choice([1,2,3], n),
        "major_vessels":    np.random.choice([0,1,2,3], n),
        "thal":             np.random.choice([3,6,7], n),
    })
    risk = (
        0.3 * (df["age"] > 55) +
        0.2 * (df["cholesterol"] > 240) +
        0.2 * (df["exercise_angina"]) +
        0.1 * (df["fasting_sugar"]) +
        0.2 * (df["major_vessels"] > 0)
    )
    df["TARGET"] = (np.random.rand(n) < risk).astype(int)
    feature_names = [c for c in df.columns if c != "TARGET"]
    return df, feature_names, "Heart Disease", ["No Disease", "Has Disease"]


# ──────────────────────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────────
def explore_data(df, target_col, disease_name, class_names):
    print(f"\n{'='*60}")
    print(f"  STEP 3: EDA — {disease_name}")
    print(f"{'='*60}")

    print(f"\n📋 Shape  : {df.shape}")
    print(f"📊 Sample :\n{df.head(3)}")
    print(f"\n📈 Stats  :\n{df.describe().round(2)}")

    label_col = "DIAGNOSIS_LABEL" if "DIAGNOSIS_LABEL" in df.columns else target_col
    counts = df[target_col].value_counts()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{disease_name} — Exploratory Analysis", fontsize=14, fontweight="bold")

    # Chart 1: Class distribution
    axes[0,0].bar(class_names, [counts.get(0,0), counts.get(1,0)],
                  color=["#e74c3c","#2ecc71"], edgecolor="white", linewidth=1.5)
    axes[0,0].set_title("Class Distribution")
    axes[0,0].set_ylabel("Count")
    for i, v in enumerate([counts.get(0,0), counts.get(1,0)]):
        axes[0,0].text(i, v+2, str(v), ha="center", fontweight="bold")

    # Chart 2,3,4,5: Numeric features distribution by class
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [target_col, "DIAGNOSIS_LABEL"]]
    for idx, col in enumerate(numeric_cols[:4]):
        ax = axes.flatten()[idx+1]
        for class_val, color, label in zip([0,1], ["#e74c3c","#2ecc71"], class_names):
            subset = df[df[target_col] == class_val][col].dropna()
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="white")
        ax.set_title(f"{col}")
        ax.legend(fontsize=7)

    # Chart 6: Correlation with target
    corr_with_target = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_with_target.sort_values().plot.barh(ax=axes[1,2], color="#3498db")
    axes[1,2].set_title(f"Feature Correlation with {target_col}")
    axes[1,2].axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    fname = f"eda_{disease_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ EDA charts saved → {fname}")


# ──────────────────────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
def feature_engineering_heart(df):
    """Add meaningful features for heart disease prediction."""
    df = df.copy()
    df["bp_per_age"]     = (df["resting_bp"] / (df["age"] + 1)).round(3)
    df["hr_reserve"]     = (220 - df["age"] - df["max_heart_rate"]).round(0)
    df["chol_risk"]      = (df["cholesterol"] > 240).astype(int)
    df["senior"]         = (df["age"] > 55).astype(int)
    print("\n✅ Heart disease features engineered: bp_per_age, hr_reserve, chol_risk, senior")
    return df


def feature_engineering_cancer(df):
    """Add ratio features for breast cancer."""
    df = df.copy()
    if "mean radius" in df.columns and "mean texture" in df.columns:
        df["radius_texture_ratio"] = (df["mean radius"] / (df["mean texture"] + 0.001)).round(3)
    if "mean area" in df.columns and "mean smoothness" in df.columns:
        df["area_smoothness"]      = (df["mean area"] * df["mean smoothness"]).round(3)
    print("\n✅ Cancer features engineered: radius_texture_ratio, area_smoothness")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 5: PREPARE DATA
# ──────────────────────────────────────────────────────────────
def prepare_data(df, target_col):
    drop_cols = [target_col, "DIAGNOSIS_LABEL"] if "DIAGNOSIS_LABEL" in df.columns else [target_col]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


# ──────────────────────────────────────────────────────────────
# STEP 6: TRAIN & EVALUATE MODELS
# ──────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test, disease_name):
    print(f"\n{'='*60}")
    print(f"  STEP 6: TRAINING MODELS — {disease_name}")
    print(f"{'='*60}")

    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                            use_label_encoder=False, eval_metric="logloss",
                            random_state=42, verbosity=0)
    else:
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                          learning_rate=0.1, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "SVM (RBF Kernel)":    SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6,
                                                       random_state=42, n_jobs=-1),
        "XGBoost":             xgb,
    }

    results = []
    for model_name, model in models.items():
        print(f"\n🔄 Training: {model_name} ...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Cross-validation gives a more reliable accuracy score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)
        cv   = cv_scores.mean()

        print(f"   ✅ Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"   ✅ Precision   : {prec:.4f}  — When it predicts disease, {prec*100:.1f}% are correct")
        print(f"   ✅ Recall      : {rec:.4f}  — Catches {rec*100:.1f}% of actual disease cases")
        print(f"   ✅ F1-Score    : {f1:.4f}  — Balanced score")
        print(f"   ✅ ROC-AUC     : {auc:.4f}  — Overall discriminating power")
        print(f"   ✅ CV-AUC(5x)  : {cv:.4f} ± {cv_scores.std():.4f}  — Cross-validated AUC")

        results.append({
            "name":    model_name,
            "model":   model,
            "acc":     acc,
            "prec":    prec,
            "rec":     rec,
            "f1":      f1,
            "auc":     auc,
            "cv_auc":  cv,
            "y_pred":  y_pred,
            "y_proba": y_proba
        })

    return results


# ──────────────────────────────────────────────────────────────
# STEP 7: VISUALIZE RESULTS
# ──────────────────────────────────────────────────────────────
def visualize_results(results, y_test, feature_names, disease_name, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"{disease_name} — Model Evaluation Dashboard", fontsize=14, fontweight="bold")

    colors  = ["#3498db","#e67e22","#2ecc71","#e74c3c"]
    metrics = ["acc","prec","rec","f1","auc"]
    mlabels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]

    # Chart 1: Metric bars
    x     = np.arange(len(metrics))
    width = 0.2
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        axes[0,0].bar(x + i*width, vals, width, label=res["name"],
                      color=colors[i], alpha=0.85)
    axes[0,0].set_xticks(x + width * 1.5)
    axes[0,0].set_xticklabels(mlabels)
    axes[0,0].set_ylim(0, 1.15)
    axes[0,0].set_title("All Metrics — All Models")
    axes[0,0].legend(fontsize=7)

    # Chart 2: ROC Curves
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0,1].plot(fpr, tpr, color=colors[i], lw=2,
                       label=f"{res['name']} (AUC={res['auc']:.3f})")
    axes[0,1].plot([0,1],[0,1],"k--",lw=1)
    axes[0,1].set_title("ROC Curves")
    axes[0,1].set_xlabel("FPR — False Positive Rate")
    axes[0,1].set_ylabel("TPR — True Positive Rate")
    axes[0,1].legend(fontsize=7); axes[0,1].grid(alpha=0.3)

    # Chart 3: Cross-Validation AUC comparison
    model_names = [r["name"] for r in results]
    cv_aucs     = [r["cv_auc"] for r in results]
    axes[0,2].barh(model_names, cv_aucs, color=colors)
    axes[0,2].set_xlim(0, 1)
    axes[0,2].set_title("Cross-Validation AUC (5-Fold)")
    axes[0,2].axvline(0.5, color="red", linestyle="--", lw=1, label="Random")
    for i, v in enumerate(cv_aucs):
        axes[0,2].text(v+0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # Chart 4,5: Confusion Matrices (best 2 models)
    sorted_res = sorted(results, key=lambda r: r["auc"], reverse=True)
    for idx, res in enumerate(sorted_res[:2]):
        ax = axes[1, idx]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5)
        ax.set_title(f"Confusion Matrix\n{res['name']}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # Chart 6: Feature Importance (Random Forest)
    rf = next((r for r in results if "Random Forest" in r["name"]), None)
    if rf:
        imp = pd.Series(rf["model"].feature_importances_, index=feature_names)
        imp.nlargest(12).sort_values().plot.barh(ax=axes[1,2], color="#3498db")
        axes[1,2].set_title("Top Features (Random Forest)")

    plt.tight_layout()
    fname = f"evaluation_{disease_name.replace(' ','_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Evaluation saved → {fname}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {disease_name}")
    print(f"{'='*60}")
    summary = pd.DataFrame([{
        "Model":    r["name"],
        "Acc":      f"{r['acc']:.3f}",
        "Prec":     f"{r['prec']:.3f}",
        "Recall":   f"{r['rec']:.3f}",
        "F1":       f"{r['f1']:.3f}",
        "AUC":      f"{r['auc']:.3f}",
        "CV-AUC":   f"{r['cv_auc']:.3f}",
    } for r in results])
    print(summary.to_string(index=False))

    best = max(results, key=lambda r: r["auc"])
    print(f"\n🏆 Best: {best['name']}  AUC={best['auc']:.4f}")
    return best


# ──────────────────────────────────────────────────────────────
# STEP 8: PREDICT FOR A NEW PATIENT
# ──────────────────────────────────────────────────────────────
def predict_patient(model, feature_names, disease_name, class_names, sample_values=None):
    print(f"\n{'='*60}")
    print(f"  STEP 8: NEW PATIENT PREDICTION — {disease_name}")
    print(f"{'='*60}")

    if sample_values is None:
        # Use median values as a demo patient
        sample_values = {f: 0.5 for f in feature_names}

    patient_df = pd.DataFrame([{f: sample_values.get(f, 0) for f in feature_names}])

    scaler = StandardScaler()
    patient_scaled = scaler.fit_transform(patient_df)

    pred  = model.predict(patient_scaled)[0]
    proba = model.predict_proba(patient_scaled)[0]

    print(f"\n🎯 PREDICTION RESULT:")
    print(f"   Predicted Class      : {class_names[pred]}")
    print(f"   Probability of {class_names[0]:<12}: {proba[0]:.2%}")
    print(f"   Probability of {class_names[1]:<12}: {proba[1]:.2%}")

    if pred == 0:
        print(f"\n   ⚠️  HIGH RISK: {class_names[0]} detected — consult a doctor.")
    else:
        print(f"\n   ✅  LOW RISK: {class_names[1]} — continue regular checkups.")


# ──────────────────────────────────────────────────────────────
# MAIN — Run both disease predictions
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── DISEASE 1: BREAST CANCER ──────────────────────────────
    print("\n" + "#"*60)
    print("#   DISEASE 1: BREAST CANCER PREDICTION")
    print("#"*60)

    df_cancer, feat_cancer, name_cancer, classes_cancer = load_breast_cancer_data()
    explore_data(df_cancer, "DIAGNOSIS", name_cancer, classes_cancer)
    df_cancer = feature_engineering_cancer(df_cancer)
    X_tr, X_te, y_tr, y_te, feats = prepare_data(df_cancer, "DIAGNOSIS")
    results_cancer = train_and_evaluate(X_tr, X_te, y_tr, y_te, name_cancer)
    best_cancer = visualize_results(results_cancer, y_te, feats, name_cancer, classes_cancer)
    predict_patient(best_cancer["model"], feats, name_cancer, classes_cancer)

    # ── DISEASE 2: HEART DISEASE ──────────────────────────────
    print("\n" + "#"*60)
    print("#   DISEASE 2: HEART DISEASE PREDICTION")
    print("#"*60)

    df_heart, feat_heart, name_heart, classes_heart = load_heart_disease_data()
    explore_data(df_heart, "TARGET", name_heart, classes_heart)
    df_heart = feature_engineering_heart(df_heart)
    X_tr, X_te, y_tr, y_te, feats = prepare_data(df_heart, "TARGET")
    results_heart = train_and_evaluate(X_tr, X_te, y_tr, y_te, name_heart)
    best_heart = visualize_results(results_heart, y_te, feats, name_heart, classes_heart)
    predict_patient(best_heart["model"], feats, name_heart, classes_heart)

    print("\n" + "="*60)
    print("  ✅  TASK 4 COMPLETE!")
    print("  Output files generated:")
    print("    → eda_breast_cancer.png")
    print("    → eda_heart_disease.png")
    print("    → evaluation_breast_cancer.png")
    print("    → evaluation_heart_disease.png")
    print("="*60)
