# 💳 Task 1: Credit Scoring Model

Predicts whether a loan applicant will **default** (fail to repay) based on their financial history.

---

## 🎯 Objective
Build a binary classification model that answers:
> **"Given someone's financial profile, will they default on their loan?"**

---

## 🧠 Concepts Covered
| Concept | What It Is |
|---|---|
| **Classification** | Predicting a category (Default / No Default) |
| **Feature Engineering** | Creating new useful columns from existing data |
| **Train/Test Split** | Splitting data so the model is tested on unseen data |
| **StandardScaler** | Normalizing numbers so all features have equal weight |
| **Logistic Regression** | Simple linear model for yes/no predictions |
| **Decision Tree** | Flowchart-like model that splits data on conditions |
| **Random Forest** | Many decision trees combined for better accuracy |
| **Precision** | Of all predicted defaults, how many were actually defaults |
| **Recall** | Of all real defaults, how many did we catch |
| **F1-Score** | Balanced average of Precision and Recall |
| **ROC-AUC** | Overall model quality score (1.0 = perfect) |

---

## 📁 Project Structure
```
Task1_Credit_Scoring/
├── credit_scoring.py     ← Main code (run this)
├── requirements.txt      ← Python packages needed
├── README.md             ← This file
└── outputs/              ← Auto-created when you run the code
    ├── eda_analysis.png
    └── model_evaluation.png
```

---

## ⚙️ Setup & Run

### Step 1 — Install Python packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run the model
```bash
python credit_scoring.py
```

### Step 3 — View outputs
After running, you'll see these files in the same folder:
- `eda_analysis.png` — Charts showing the dataset
- `model_evaluation.png` — ROC curves, confusion matrix, feature importances

---

## 📊 Features Used
| Feature | Description |
|---|---|
| `age` | Applicant's age |
| `income` | Monthly income ($) |
| `loan_amount` | Requested loan size ($) |
| `credit_history` | Years of credit history |
| `num_late_payments` | Times paid late in the past |
| `debt_ratio` | Fraction of income going to debt payments |
| `num_credit_cards` | Number of active credit cards |
| `employment_years` | Years at current employer |
| `loan_to_income` | *(engineered)* Loan amount ÷ income |
| `risk_score` | *(engineered)* Custom composite risk metric |

---

## 📈 Expected Results
| Model | Typical AUC |
|---|---|
| Logistic Regression | ~0.72–0.78 |
| Decision Tree | ~0.70–0.75 |
| Random Forest | ~0.78–0.85 |

---

## 🔄 Using Real Kaggle Data
1. Go to: https://www.kaggle.com/c/GiveMeSomeCredit/data
2. Download `cs-training.csv`
3. Place it in this folder
4. In `credit_scoring.py`, change:
```python
df = create_dataset(n_samples=2000)
```
to:
```python
df = pd.read_csv("cs-training.csv", index_col=0)
```

---

## 📚 Resources
- [Scikit-learn Docs](https://scikit-learn.org/stable/)
- [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- [Logistic Regression Explained](https://towardsdatascience.com/logistic-regression-explained)
