# 🏥 Task 4: Disease Prediction from Medical Data

Predicts whether a patient has **Breast Cancer** (malignant/benign) or **Heart Disease** using clinical data.

---

## 🎯 Objective
Build classification models that answer:
> **"Based on a patient's medical data, do they have this disease?"**

---

## 🧠 Concepts Covered
| Concept | What It Is |
|---|---|
| **Binary Classification** | Two-class prediction (disease / no disease) |
| **SVM** | Support Vector Machine — finds the best boundary between classes |
| **XGBoost** | Powerful gradient boosting algorithm, often wins Kaggle competitions |
| **Cross-Validation** | Tests model on 5 different splits to get reliable accuracy |
| **Feature Importance** | Which medical measurements matter most |
| **Confusion Matrix** | Table showing correct vs. incorrect predictions |
| **Recall (Sensitivity)** | Critical in medicine — catching all real disease cases |
| **Precision** | Avoiding false alarms |

---

## 📁 Project Structure
```
Task4_Disease_Prediction/
├── disease_prediction.py  ← Main code (run this)
├── requirements.txt       ← Python packages needed
├── README.md              ← This file
└── outputs/               ← Auto-created after running
    ├── eda_breast_cancer.png
    ├── eda_heart_disease.png
    ├── evaluation_breast_cancer.png
    └── evaluation_heart_disease.png
```

---

## ⚙️ Setup & Run

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Run
```bash
python disease_prediction.py
```
> No data download needed! Datasets load automatically from sklearn and UCI.

---

## 📊 Datasets Used

### 1. Breast Cancer Wisconsin (built into sklearn)
- **569 patients**, **30 features**
- Features: cell radius, texture, perimeter, area, smoothness etc.
- Target: **Malignant (0)** or **Benign (1)**

### 2. Heart Disease (Cleveland, UCI)
- **303 patients**, **13 features**
- Features: age, chest pain type, cholesterol, max heart rate, etc.
- Target: **No Disease (0)** or **Has Disease (1)**

---

## 📈 Expected Results

### Breast Cancer
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | ~94–96% |
| SVM | ~95–97% |
| Random Forest | ~95–97% |
| XGBoost | ~96–98% |

### Heart Disease
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | ~80–83% |
| SVM | ~82–85% |
| Random Forest | ~82–86% |
| XGBoost | ~83–87% |

---

## ⚠️ Why Recall Matters in Medicine
In disease prediction, **missing a real case (false negative) is dangerous**.

- A false negative = telling a sick person they're healthy ❌
- Therefore we **prioritize Recall / Sensitivity** over Precision

---

## 📚 Resources
- [Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Heart Disease Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Understanding Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix)
