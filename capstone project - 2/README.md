# Capstone Project Part 2: Supervised Machine Learning Model

This section of the project focuses on training and evaluating supervised machine learning models for both regression (predicting Fare) and classification (predicting Survived).

## Preprocessing Decisions

### Categorical Encoding
- **Ordinal Variables (Pclass)**: Pclass (1, 2, 3) has a natural order (1st class is higher/more expensive than 3rd class). We left it as a numeric ordinal variable. 
- **Nominal Variables (Sex, Embarked)**: These features have no intrinsic numerical order (e.g., 'male' is not mathematically greater than 'female'). We applied one-hot encoding (`pd.get_dummies(drop_first=True)`). Using label encoding here would artificially impose a false ordinal relationship, causing the model to interpret one category as "greater" than another. Dropping the first dummy column prevents multicollinearity (the dummy variable trap).

### Leak-Free Scaling
To prevent **data leakage**, we split the dataset into training and test sets *before* applying the `StandardScaler`. 
The scaler was fitted **only on the training features** (`scaler.fit(X_train)`), and then used to transform both the train and test sets. If we had fitted the scaler on the entire dataset, information about the test set's distribution (its mean and variance) would "leak" into the training process, leading to overly optimistic performance estimates.

---

## Regression Model: Predicting Fare

### Evaluation Metrics
We trained an Ordinary Least Squares (OLS) Linear Regression model and a Ridge Regression model (alpha=1.0).

| Model | Mean Squared Error (MSE) | R² Score |
| :--- | :--- | :--- |
| **Linear Regression (OLS)** | 1752.16 | 0.2641 |
| **Ridge Regression (alpha=1.0)** | 1751.96 | 0.2642 |

### Coefficient Interpretation
The top three features with the largest absolute coefficients in the OLS model are:
1. **Pclass (-24.58)**: A large negative coefficient. For every one-unit increase in the scaled `Pclass` feature (e.g., moving from 1st to 2nd or 3rd class), the predicted Fare drops significantly by 24.58 units.
2. **Parch (7.59)**: A positive coefficient. An increase in the scaled number of parents/children aboard is associated with a 7.59 unit increase in Fare.
3. **SibSp (6.53)**: A positive coefficient. An increase in the scaled number of siblings/spouses aboard is associated with a 6.53 unit increase in Fare.

### Ridge vs. OLS Comparison
Ridge Regression introduces an L2 penalty on the size of the coefficients, controlled by the `alpha` parameter. While OLS can produce excessively large coefficients to perfectly fit the training data (leading to overfitting, especially with multicollinearity), Ridge shrinks these coefficients toward zero. In our dataset, the performance of Ridge (R² = 0.2642) is nearly identical to OLS (R² = 0.2641), suggesting that multicollinearity or coefficient explosion wasn't a severe issue here, but Ridge still provided a microscopic improvement in generalization.

---

## Classification Model: Predicting Survival

### Class Imbalance Handling
The training set exhibited a mild imbalance:
- **Did not survive (0)**: 440 samples
- **Survived (1)**: 271 samples

Instead of artificially generating synthetic data (e.g., SMOTE), we handled this using `class_weight='balanced'` in the Logistic Regression constructor. This inversely weights the loss function based on class frequencies, effectively giving the minority class (Survived) more importance during training without altering the raw row counts.

### Baseline Evaluation (Threshold = 0.5)

**Confusion Matrix:**
```
[[83 26]
 [12 57]]
```

**Classification Report:**
```
              precision    recall  f1-score   support
           0       0.87      0.76      0.81       109
           1       0.69      0.83      0.75        69
    accuracy                           0.79       178
   macro avg       0.78      0.79      0.78       178
weighted avg       0.80      0.79      0.79       178
```

- **AUC Value**: **0.8492** (The ROC curve is saved as `roc_curve.png`).
- **AUC Interpretation**: The Area Under the Curve (0.8492) indicates that if we randomly pick one passenger who survived and one who didn't, there is an 84.9% chance the model will assign a higher survival probability to the one who actually survived. It measures the model's ability to separate the two classes across all possible thresholds.

### Precision and Recall Formulas
- **Precision** = $TP / (TP + FP)$
- **Recall** = $TP / (TP + FN)$

*(Where TP = True Positives, FP = False Positives, FN = False Negatives)*

**Which is more important?**
In the context of Titanic survival (a disaster scenario), **Recall** is arguably more important. A false negative means predicting someone will die when they actually survive. If this model were used for rescue prioritization, a false negative might mean abandoning someone who could be saved. Therefore, capturing as many true survivors as possible (Recall) takes precedence over ensuring every predicted survivor is perfectly correct (Precision).

---

## Decision-Threshold Sensitivity

By default, logistic regression uses a 0.5 threshold. We varied this from 0.30 to 0.70 to observe the tradeoff.

| Threshold | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| 0.30 | 0.5701 | 0.8841 | 0.6932 |
| 0.40 | 0.6277 | 0.8551 | 0.7239 |
| **0.50** | **0.6867** | **0.8261** | **0.7500** |
| 0.60 | 0.6923 | 0.7826 | 0.7347 |
| 0.70 | 0.7692 | 0.7246 | 0.7463 |

- **Best F1-Score**: The default threshold of **0.50** maximizes the F1-score (0.7500).
- **Threshold Adjustment**: Since we established that **Recall** is more important for this domain, we would logically **lower the threshold** (e.g., to 0.30 or 0.40). Lowering the threshold increases Recall (catching more survivors) at the direct cost of Precision (increasing false positives/false alarms).

---

## Regularization Experiment & Bootstrapping

We trained a second model with stronger regularization (`C=0.01`). 
- **C Parameter Explanation**: In Logistic Regression, `C` is the inverse of regularization strength. A smaller `C` (like 0.01) means *stronger* penalty on large coefficients, creating a simpler model that resists overfitting but might underfit. A larger `C` (like 1.0) applies less penalty. 

| Model | Precision | Recall | AUC |
| :--- | :--- | :--- | :--- |
| **C = 1.0 (Baseline)** | 0.6867 | 0.8261 | 0.8492 |
| **C = 0.01 (Stronger)** | 0.6829 | 0.8116 | 0.8521 |

Reducing `C` slightly improved the AUC (from 0.8492 to 0.8521), indicating that stronger regularization marginally helped the model's overall ranking ability on this dataset, though it slightly lowered precision and recall at the 0.5 threshold.

### Bootstrapping Analysis
To quantify if the baseline (C=1.0) model is reliably different from the heavily regularized (C=0.01) model, we calculated the AUC difference across 500 bootstrap test samples.

- **Mean AUC Difference (C=1.0 - C=0.01)**: -0.0029
- **95% Confidence Interval**: [-0.0138, 0.0084]

**Conclusion**: The 95% confidence interval **includes zero**. This means the difference in AUC between the two models is not statistically reliable across different data samples. Neither model definitively outperforms the other, so we could safely choose either.
