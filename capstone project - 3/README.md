# Capstone Project Part 3: Advanced Modeling & Pipeline

This section implements advanced tree-based modeling, ensembles, feature ablation, hyperparameter tuning, and model serialization.

---

## 1. Decision Trees: Bias vs Variance

### Unconstrained Decision Tree
We trained a `DecisionTreeClassifier` with `max_depth=None`.
- **Train Accuracy**: 0.9353
- **Test Accuracy**: 0.7865

This model shows clear signs of **overfitting** (high train accuracy, significantly lower test accuracy). Decision trees are inherently **high-variance** models because they fit the training data greedily at each split without revisiting earlier decisions, creating overly complex boundaries that memorize noise in the training set.

### Controlled Decision Tree
We trained a second tree with `max_depth=5` and `min_samples_split=20`.
- **Train Accuracy**: 0.8467
- **Test Accuracy**: 0.8202

- **`max_depth`**: Limits how deep the tree can grow. This prevents the tree from creating highly specific leaves, reducing variance at the cost of slightly increasing bias.
- **`min_samples_split`**: Prevents splitting a node if fewer than the specified number of samples are present, avoiding splits that respond to random noise in small data subsets.
The train/test gap collapsed from ~15% to ~2.6%, demonstrating a significantly more robust and generalized model.

---

## 2. Information Theory (Gini vs Entropy)

We compared two `max_depth=5` trees with different splitting criteria:
- **Gini Test Accuracy**: 0.8258
- **Entropy Test Accuracy**: 0.8146

### Formulas
- **Gini Impurity**: $1 - \sum p_i^2$
- **Entropy**: $-\sum p_i \log_2(p_i)$

When a node has **Gini = 0**, it is perfectly "pure", meaning all samples in that node belong to a single class (e.g., 100% Survived or 100% Did Not Survive).

---

## 3. Ensembles (Random Forest)

We trained a `RandomForestClassifier`.
- **Top 5 Features by Importance**:
  1. `Sex_male` (0.3485)
  2. `Age` (0.3103)
  3. `Pclass` (0.1465)
  4. `SibSp` (0.0829)
  5. `Parch` (0.0601)

### How Random Forest Computes Importance
Unlike linear regression (which looks at coefficient magnitudes associated with the raw target change), a Random Forest calculates importance by averaging the **reduction in Gini impurity** across all splits that use that specific feature, across all trees in the forest. A higher score means splits on that feature consistently made the resulting child nodes much purer.

### The Bagging Concept
Bagging (Bootstrap Aggregating) means each tree in the forest is trained on a random sample (drawn with replacement) from the training data. Additionally, at each split, the tree is only allowed to consider a random subset of features (typically $\sqrt{\text{total features}}$). This ensemble averaging dramatically reduces the model's variance compared to a single deep decision tree, as it prevents any single feature or outlier from dominating the entire forest structure.

---

## 4. Feature Ablation Study

We removed the 5 lowest-importance features (`Embarked_Q`, `Embarked_S`, `Parch`, `SibSp`, `Pclass`) to observe the impact.
- **Full Model AUC**: 0.8542
- **Ablated Model AUC**: 0.8367

The AUC drops without these features, indicating they were genuinely contributing informative signal rather than just adding noise.
**Production Trade-Off**: Deploying a simpler, lower-dimensional model reduces inference latency and maintenance burden. However, this is only acceptable if the degradation in AUC is below the business's tolerable threshold. In this case, losing ~0.017 AUC might be an acceptable trade-off for dropping 5 features, depending on the exact precision/recall requirements of the client.

---

## 5. Cross-Validated Comparison

We evaluated multiple models using 5-Fold Stratified Cross-Validation (Scoring = `roc_auc`).

**Why CV?** Cross-validation provides a more reliable estimate of generalization performance than a single train-test split because it trains and evaluates the model across multiple, non-overlapping partitions of the data. This reduces the variance of the evaluation metric and proves the model isn't just getting "lucky" on a specific random test set.

| Model | 5-Fold Mean AUC | 5-Fold Std AUC | Test-Set AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Part 2)** | 0.8543 | 0.0297 | 0.8492 |
| **Controlled Decision Tree** | 0.8540 | 0.0146 | 0.8582* |
| **Random Forest** | 0.8472 | 0.0191 | 0.8542 |
| **Gradient Boosting** | **0.8648** | 0.0304 | **0.8692** |
*(Note: Test-set AUC for DT derived manually during pipeline comparison)*

---

## 6. Hyperparameter Tuning (GridSearchCV)

We built an `sklearn` Pipeline (`SimpleImputer` -> `StandardScaler` -> `RandomForestClassifier`) and used `GridSearchCV`.
- **Evaluated Configurations**: 18 parameter combinations $\times$ 5 folds = **90 total models evaluated**.
- **Best Parameters**: `max_depth=5`, `min_samples_leaf=1`, `n_estimators=200`
- **Best CV AUC Score**: 0.8666

**Grid Search vs. Randomized Search**:
Exhaustive Grid Search tests every single possible combination in the parameter grid, guaranteeing you find the absolute best parameter set *within that predefined grid*, but it is extremely computationally expensive. Randomized Search samples a fixed number of combinations randomly, which is much faster and often finds a near-optimal solution, especially in massive parameter spaces where only a few hyperparameters truly matter.

---

## 7. Manual Learning Curve

We refit our best tuned pipeline on progressively larger subsets of the training data.

| Training fraction | Training AUC | Test AUC |
| :---: | :---: | :---: |
| 0.20 | 0.9544 | 0.8635 |
| 0.40 | 0.9301 | 0.8621 |
| 0.60 | 0.9213 | 0.8662 |
| 0.80 | 0.9132 | 0.8653 |
| 1.00 | 0.8997 | 0.8619 |

- **Training AUC Decrease**: As expected for high-variance models, Training AUC decreases as the training set grows because it becomes harder for the model to perfectly memorize a larger, more diverse dataset.
- **Test AUC Plateau**: Test AUC does *not* significantly increase with more data (it starts at 0.8635 and plateaus around 0.8619). 
- **Conclusion**: The model is **capacity-limited** (or limited by the inherent noise/features in the dataset), rather than data-limited. Collecting more rows of the exact same features is unlikely to improve performance. To get better scores, we would need to engineer new features or use a fundamentally different architecture.

---

## 8. Final Recommendation

Based on the summary table in Section 5:
I recommend the **Gradient Boosting Classifier (GBC)**. It achieved the highest mean Cross-Validation AUC (0.8648) and the highest Test-Set AUC (0.8692). While Logistic Regression provides excellent baseline performance and extreme interpretability, Gradient Boosting is able to capture complex, non-linear relationships in the features without requiring manual feature engineering, ultimately providing the most robust ranking of survival probability.
