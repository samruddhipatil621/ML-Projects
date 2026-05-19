# =============================================================
#   TASK 1: CREDIT SCORING MODEL
#   Predict whether a person will default on a loan (Yes/No)
#   Dataset: Synthetic (runs instantly, no download needed)
#   Author: Add your name here
# =============================================================

# ──────────────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# These are Python packages that give us ready-made ML tools
# ──────────────────────────────────────────────────────────────
import numpy as np                          # For number operations
import pandas as pd                         # For working with tables (DataFrames)
import matplotlib.pyplot as plt             # For drawing charts
import seaborn as sns                       # For pretty charts
import warnings
warnings.filterwarnings("ignore")           # Hide unimportant warnings

# Scikit-learn: the main ML library we will use
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


# ──────────────────────────────────────────────────────────────
# STEP 2: CREATE / LOAD DATASET
# We create a realistic synthetic credit dataset.
# Features explained:
#   age               → Person's age in years
#   income            → Monthly income in dollars
#   loan_amount       → Loan they are requesting
#   credit_history    → Years of credit history
#   num_late_payments → Times they paid late in past
#   debt_ratio        → What fraction of income goes to debts (0–1)
#   num_credit_cards  → How many credit cards they have
#   employment_years  → Years at current job
#   DEFAULT           → 1 = will default, 0 = will repay (this is what we predict)
# ──────────────────────────────────────────────────────────────
def create_dataset(n_samples=2000, random_state=42):
    """
    Generate a synthetic credit dataset.
    In a real project, replace this with:
        df = pd.read_csv("your_data.csv")
    """
    np.random.seed(random_state)
    n = n_samples

    age               = np.random.randint(22, 70, n)
    income            = np.random.normal(5000, 2000, n).clip(1000, 20000).round(0)
    loan_amount       = np.random.normal(15000, 8000, n).clip(1000, 50000).round(0)
    credit_history    = np.random.randint(0, 20, n)
    num_late_payments = np.random.choice([0,1,2,3,4,5], n, p=[0.60,0.20,0.09,0.05,0.04,0.02])
    debt_ratio        = np.random.beta(2, 5, n).round(3)
    num_credit_cards  = np.random.randint(0, 8, n)
    employment_years  = np.random.randint(0, 30, n)

    # Create default probability based on features (realistic logic)
    default_prob = (
        0.05                                        # base rate
        + 0.15 * (num_late_payments / 5)            # more late payments → higher risk
        + 0.10 * (debt_ratio)                       # higher debt ratio → higher risk
        + 0.05 * (loan_amount / income / 10)        # large loan vs income → higher risk
        - 0.05 * (credit_history / 20)              # longer history → lower risk
        - 0.03 * (employment_years / 30)            # stable job → lower risk
        - 0.02 * (income / 20000)                   # higher income → lower risk
    ).clip(0, 0.9)

    default = (np.random.rand(n) < default_prob).astype(int)

    df = pd.DataFrame({
        "age":               age,
        "income":            income,
        "loan_amount":       loan_amount,
        "credit_history":    credit_history,
        "num_late_payments": num_late_payments,
        "debt_ratio":        debt_ratio,
        "num_credit_cards":  num_credit_cards,
        "employment_years":  employment_years,
        "DEFAULT":           default
    })

    return df


# ──────────────────────────────────────────────────────────────
# STEP 3: EXPLORE THE DATA (EDA)
# Always look at your data before training any model!
# ──────────────────────────────────────────────────────────────
def explore_data(df):
    print("\n" + "="*60)
    print("  STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    print(f"\n📋 Dataset shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📊 First 5 rows:\n{df.head()}")
    print(f"\n📈 Basic Statistics:\n{df.describe().round(2)}")

    # How many people defaulted vs did not?
    counts = df["DEFAULT"].value_counts()
    print(f"\n⚖️  Class Distribution:")
    print(f"   Did NOT default (0): {counts[0]} ({counts[0]/len(df)*100:.1f}%)")
    print(f"   DID default    (1): {counts[1]} ({counts[1]/len(df)*100:.1f}%)")

    print(f"\n🔍 Missing values  : {df.isnull().sum().sum()} (none — our data is clean)")

    # ── Save EDA charts ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Credit Dataset — Exploratory Analysis", fontsize=14, fontweight="bold")

    # Chart 1: Default distribution
    axes[0,0].bar(["No Default (0)", "Default (1)"], [counts[0], counts[1]],
                  color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
    axes[0,0].set_title("Target Class Distribution")
    axes[0,0].set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0,0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    # Chart 2: Age distribution
    df["age"].hist(ax=axes[0,1], bins=25, color="#3498db", edgecolor="white")
    axes[0,1].set_title("Age Distribution")
    axes[0,1].set_xlabel("Age")

    # Chart 3: Income distribution
    df["income"].hist(ax=axes[0,2], bins=25, color="#9b59b6", edgecolor="white")
    axes[0,2].set_title("Income Distribution")
    axes[0,2].set_xlabel("Monthly Income ($)")

    # Chart 4: Late payments vs Default rate
    late_default = df.groupby("num_late_payments")["DEFAULT"].mean() * 100
    axes[1,0].bar(late_default.index, late_default.values, color="#e67e22", edgecolor="white")
    axes[1,0].set_title("Late Payments vs Default Rate")
    axes[1,0].set_xlabel("Number of Late Payments")
    axes[1,0].set_ylabel("Default Rate (%)")

    # Chart 5: Debt Ratio vs Default
    df.boxplot(column="debt_ratio", by="DEFAULT", ax=axes[1,1], patch_artist=True)
    axes[1,1].set_title("Debt Ratio by Default Status")
    axes[1,1].set_xlabel("Default (0=No, 1=Yes)")
    plt.sca(axes[1,1])
    plt.title("Debt Ratio by Default Status")

    # Chart 6: Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1,2],
                square=True, cbar_kws={"shrink": 0.8})
    axes[1,2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("eda_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ EDA charts saved  → eda_analysis.png")


# ──────────────────────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# Create new useful features from existing ones
# ──────────────────────────────────────────────────────────────
def feature_engineering(df):
    print("\n" + "="*60)
    print("  STEP 4: FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()

    # New Feature 1: Loan-to-Income ratio (how big is the loan vs their income?)
    df["loan_to_income"] = (df["loan_amount"] / (df["income"] + 1)).round(3)

    # New Feature 2: Risk score (custom formula combining risk factors)
    df["risk_score"] = (
        df["num_late_payments"] * 2 +
        df["debt_ratio"] * 3 -
        df["credit_history"] * 0.5 -
        df["employment_years"] * 0.3
    ).round(3)

    # New Feature 3: Is the loan amount very large? (binary yes/no)
    df["high_loan_flag"] = (df["loan_to_income"] > 3).astype(int)

    # New Feature 4: Is the person financially stable? (binary yes/no)
    df["stable_flag"] = (
        (df["employment_years"] >= 3) &
        (df["num_late_payments"] == 0) &
        (df["debt_ratio"] < 0.4)
    ).astype(int)

    new_features = ["loan_to_income", "risk_score", "high_loan_flag", "stable_flag"]
    print(f"\n✅ Added {len(new_features)} new features: {new_features}")
    print(f"   Total features now: {df.shape[1] - 1}")

    return df


# ──────────────────────────────────────────────────────────────
# STEP 5: PREPARE DATA FOR TRAINING
# Split into train/test sets and scale the numbers
# ──────────────────────────────────────────────────────────────
def prepare_data(df):
    print("\n" + "="*60)
    print("  STEP 5: PREPARING DATA FOR TRAINING")
    print("="*60)

    # Separate features (X) from the target label (y)
    X = df.drop(columns=["DEFAULT"])   # Everything except what we predict
    y = df["DEFAULT"]                  # What we want to predict

    # Split: 80% for training, 20% for testing
    # stratify=y ensures both splits have similar DEFAULT ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📦 Training set    : {X_train.shape[0]} samples")
    print(f"📦 Test set        : {X_test.shape[0]} samples")
    print(f"📦 Features        : {X_train.shape[1]}")

    # Scale features: bring all numbers to a similar range (mean=0, std=1)
    # This helps models like Logistic Regression work better
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit on train, transform train
    X_test_scaled  = scaler.transform(X_test)        # Only transform test (NOT fit again)

    print(f"\n✅ Features scaled using StandardScaler")
    print(f"   (All values normalized to mean≈0, std≈1)")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


# ──────────────────────────────────────────────────────────────
# STEP 6: TRAIN & EVALUATE MODELS
# We train 3 models and compare their performance
# ──────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  STEP 6: TRAINING MODELS")
    print("="*60)

    # Define three models:
    # 1. Logistic Regression  → Simple, fast, great baseline
    # 2. Decision Tree        → Easy to understand, like a flowchart
    # 3. Random Forest        → Many trees combined = more accurate
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=6,
                                                       random_state=42, n_jobs=-1),
    }

    results = []

    for model_name, model in models.items():
        print(f"\n🔄 Training: {model_name} ...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred  = model.predict(X_test)                    # 0 or 1 predictions
        y_proba = model.predict_proba(X_test)[:, 1]       # Probability of default

        # Calculate metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_proba)

        print(f"   ✅ Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"   ✅ Precision : {prec:.4f}  (Of all predicted defaults, {prec*100:.1f}% were correct)")
        print(f"   ✅ Recall    : {rec:.4f}  (Of all real defaults, caught {rec*100:.1f}%)")
        print(f"   ✅ F1-Score  : {f1:.4f}  (Balance of Precision & Recall)")
        print(f"   ✅ ROC-AUC   : {auc:.4f}  (Overall model quality, 1.0 = perfect)")

        results.append({
            "name":    model_name,
            "model":   model,
            "acc":     acc,
            "prec":    prec,
            "rec":     rec,
            "f1":      f1,
            "auc":     auc,
            "y_pred":  y_pred,
            "y_proba": y_proba
        })

    return results


# ──────────────────────────────────────────────────────────────
# STEP 7: VISUALIZE RESULTS
# Charts to understand model performance
# ──────────────────────────────────────────────────────────────
def visualize_results(results, y_test, feature_names):
    print("\n" + "="*60)
    print("  STEP 7: VISUALIZING RESULTS")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Credit Scoring — Model Evaluation Dashboard", fontsize=15, fontweight="bold")

    colors  = ["#3498db", "#e67e22", "#2ecc71"]
    metrics = ["acc", "prec", "rec", "f1", "auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    # ── Chart 1: Metrics Comparison Bar Chart ──
    x     = np.arange(len(metrics))
    width = 0.25
    for i, res in enumerate(results):
        vals = [res[m] for m in metrics]
        axes[0,0].bar(x + i*width, vals, width, label=res["name"], color=colors[i], alpha=0.85)
    axes[0,0].set_xticks(x + width)
    axes[0,0].set_xticklabels(labels)
    axes[0,0].set_ylim(0, 1.1)
    axes[0,0].set_title("All Metrics Comparison")
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel("Score")

    # ── Chart 2: ROC Curves ──
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        axes[0,1].plot(fpr, tpr, color=colors[i], lw=2,
                       label=f"{res['name']} (AUC={res['auc']:.3f})")
    axes[0,1].plot([0,1],[0,1], "k--", lw=1, label="Random (AUC=0.5)")
    axes[0,1].set_title("ROC Curves")
    axes[0,1].set_xlabel("False Positive Rate (FPR)")
    axes[0,1].set_ylabel("True Positive Rate (TPR)")
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(alpha=0.3)

    # ── Chart 3,4,5: Confusion Matrices for each model ──
    cm_axes = [axes[0,2], axes[1,0], axes[1,1]]
    for i, res in enumerate(results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=cm_axes[i],
                    xticklabels=["No Default","Default"],
                    yticklabels=["No Default","Default"],
                    linewidths=0.5)
        cm_axes[i].set_title(f"Confusion Matrix\n{res['name']}")
        cm_axes[i].set_xlabel("Predicted")
        cm_axes[i].set_ylabel("Actual")

    # ── Chart 6: Feature Importance (Random Forest) ──
    rf = next(r for r in results if "Random Forest" in r["name"])["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(10)
    importances.plot.barh(ax=axes[1,2], color="#3498db")
    axes[1,2].set_title("Feature Importances (Random Forest)")
    axes[1,2].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Evaluation charts saved → model_evaluation.png")

    # ── Summary Table ──
    print("\n" + "="*60)
    print("  FINAL MODEL COMPARISON TABLE")
    print("="*60)
    summary = pd.DataFrame([{
        "Model":     r["name"],
        "Accuracy":  f"{r['acc']:.4f}",
        "Precision": f"{r['prec']:.4f}",
        "Recall":    f"{r['rec']:.4f}",
        "F1-Score":  f"{r['f1']:.4f}",
        "ROC-AUC":   f"{r['auc']:.4f}",
    } for r in results])
    print(summary.to_string(index=False))

    best = max(results, key=lambda r: r["auc"])
    print(f"\n🏆 Best Model: {best['name']}  (ROC-AUC = {best['auc']:.4f})")
    return best


# ──────────────────────────────────────────────────────────────
# STEP 8: PREDICT FOR A NEW PERSON
# Give it someone's details → it predicts default risk
# ──────────────────────────────────────────────────────────────
def predict_new_person(model, feature_names):
    print("\n" + "="*60)
    print("  STEP 8: PREDICT FOR A NEW PERSON")
    print("="*60)

    # Example: a 35-year-old applying for a loan
    new_person = {
        "age":               35,
        "income":            4500,
        "loan_amount":       12000,
        "credit_history":    8,
        "num_late_payments": 1,
        "debt_ratio":        0.35,
        "num_credit_cards":  2,
        "employment_years":  5,
        # Engineered features
        "loan_to_income":    12000 / (4500 + 1),
        "risk_score":        1*2 + 0.35*3 - 8*0.5 - 5*0.3,
        "high_loan_flag":    int(12000/(4500+1) > 3),
        "stable_flag":       int(5 >= 3 and 1 == 0 and 0.35 < 0.4),
    }

    # Make sure order matches training features
    person_df = pd.DataFrame([new_person])[feature_names]

    scaler = StandardScaler()  # NOTE: In production, save and reload the fitted scaler
    person_scaled = scaler.fit_transform(person_df)

    pred  = model.predict(person_scaled)[0]
    proba = model.predict_proba(person_scaled)[0][1]

    print(f"\n👤 Person Details:")
    for k, v in new_person.items():
        print(f"   {k:<25}: {v}")

    print(f"\n🎯 PREDICTION RESULT:")
    print(f"   Default Risk       : {'⚠️  HIGH RISK — LIKELY TO DEFAULT' if pred == 1 else '✅  LOW RISK — LIKELY TO REPAY'}")
    print(f"   Default Probability: {proba:.2%}")


# ──────────────────────────────────────────────────────────────
# MAIN — Run everything in order
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("   TASK 1: CREDIT SCORING MODEL")
    print("   Predicting Loan Default Risk")
    print("="*60)

    # Run all steps in sequence
    df = create_dataset(n_samples=2000)
    explore_data(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model = visualize_results(results, y_test, feature_names)
    predict_new_person(best_model["model"], feature_names)

    print("\n" + "="*60)
    print("  ✅  TASK 1 COMPLETE!")
    print("  Check these output files:")
    print("    → eda_analysis.png      (EDA charts)")
    print("    → model_evaluation.png  (Results dashboard)")
    print("="*60)
