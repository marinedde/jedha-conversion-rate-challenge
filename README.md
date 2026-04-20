# Conversion Rate Challenge — datascienceweekly.org

> **Jedha Fullstack Data Science** — Supervised Machine Learning  
> Certification CDSD · Classification · Imbalanced Dataset · Kaggle-style Competition

---

## Project Overview

datascienceweekly.org wants to predict whether a website visitor will **subscribe to their newsletter**, using a few behavioral variables. This is a **Kaggle-style competition**: models are evaluated on a held-out test set by the instructor.

This is a **supervised binary classification** task with a critical challenge: **class imbalance** (only 3.23% of visitors convert).

**Official metric: F1-score** (accuracy is misleading on imbalanced data)

---

## Objectives

| Step | Description |
|------|-------------|
| Part 1 | EDA + preprocessing + logistic regression baseline |
| Part 2 | Improve F1-score (Random Forest + GridSearchCV) |
| Part 3 | Predict on test file → submit CSV to leaderboard |
| Part 4 | Analyze model parameters → business recommendations |

---

## Dataset

**284,580 observations** · 5 features · no missing values

| Variable | Type | Description |
|----------|------|-------------|
| `country` | Categorical | Visitor country (China, UK, Germany, US) |
| `age` | Numerical | Visitor age (years) |
| `new_user` | Binary | 1 = new visitor, 0 = returning visitor |
| `source` | Categorical | Traffic source (Ads, Seo, Direct) |
| `total_pages_visited` | Numerical | Number of pages visited |
| `converted` | **Target** | 1 = subscribed, 0 = did not subscribe |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Not converted (0) | 275,400 | 96.77% |
| **Converted (1)** | **9,180** | **3.23%** |

> A model always predicting 0 would achieve 96.77% accuracy — but F1 = 0. This is why the official metric is **F1-score**.

---

## Methodology

### Preprocessing
- Outlier removal on `age` using ±3σ rule (age=123 is impossible)
- Train/test split with **`stratify=y`** — mandatory to maintain the 3.23% proportion in both sets
- **scikit-learn Pipeline**: `SimpleImputer(median)` + `StandardScaler` for numericals, `OneHotEncoder` for categoricals
- **`class_weight='balanced'`** in all models — automatically weights the 9,000 positives against 275,000 negatives

### Models

| Model | Precision | Recall | **F1 Test** |
|-------|-----------|--------|-------------|
| LogisticRegression (balanced) | 0.35 | 0.94 | 0.508 |
| RandomForest 100 trees (balanced) | 0.44 | 0.84 | 0.578 |
| **RandomForest GridSearch ✓** | **0.43** | **0.90** | **0.582** |

**Best model parameters:** `max_depth=20`, `min_samples_leaf=5`, `n_estimators=100`

### Confusion Matrix (best model on test set)

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | 52,687 ✓ | 2,191 ✗ |
| **Actual 1** | 184 ✗ | 1,651 ✓ |

---

## Key Findings

**1. `total_pages_visited` = 76% of Gini importance**  
This single variable carries most of the predictive signal. Users who visit many pages convert massively.

| Pages visited | Conversion rate |
|---------------|-----------------|
| 1–3 | ~1% |
| 4–7 | ~3% |
| 8–15 | ~15% |
| 16+ | ~50%+ |

**2. Returning users convert 5× more** (7.2% vs 1.4%)  
Retention strategies outperform acquisition campaigns.

**3. Strong geographic disparity**  
Germany: 6.3% · UK: 5.3% · US: 3.8% · China: 0.13%

**4. Traffic source has minimal impact** (< 1% importance)  
Optimizing acquisition channel is secondary to improving on-site engagement.

---

## Business Recommendations

1. **Invest in content quality and navigation** — more pages visited = more conversions. Suggested articles, internal linking, and reading time optimization are high-leverage actions.
2. **Focus on returning visitor retention** — email reminders, retargeting campaigns, and push notifications for past visitors are 5× more effective than new visitor acquisition.
3. **Concentrate marketing budgets on Germany and UK** — these markets show 5–6% conversion rates vs near-zero in China.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/jedha-conversion-rate-challenge.git
cd jedha-conversion-rate-challenge

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook Conversion_Rate_Jedha_v2.ipynb
```

Or open directly in **Google Colab** → File → Upload notebook

> GridSearchCV cell takes ~5–10 minutes to run (54 fits on 227K rows). You can reduce the grid or skip it and use the Random Forest baseline directly.

---

## File Structure

```
jedha-conversion-rate-challenge/
│
├── Conversion_Rate_Jedha_v2.ipynb                      # Main notebook (Jedha template structure)
├── conversion_data_train.csv                           # Labeled training data
├── conversion_data_test.csv                            # Unlabeled test data (for submission)
├── conversion_data_test_predictions_RF-GridSearch.csv  # Submission file
└── README.md
```

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![pandas](https://img.shields.io/badge/pandas-2.0-green)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange)

---

## Context

Part of the **Jedha Fullstack Data Science** certification (CDSD).  
Topic: Supervised Machine Learning — Classification on imbalanced data.  
Challenge format: Kaggle-style competition with F1-score leaderboard.

