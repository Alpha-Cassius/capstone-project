import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_score, recall_score, f1_score
)

def main():
    # 1. Load Data
    print("Loading data...")
    # Dynamically resolve paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(workspace_root, 'capstone project - 1', 'data', 'processed', 'cleaned_data.csv')
    df = pd.read_csv(data_path)

    # Drop unused/high-cardinality columns
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Drop any rows with remaining missing values to be safe
    df = df.dropna().reset_index(drop=True)

    # Define targets and features
    y_reg = df['Fare']
    y_clf = df['Survived']
    X_raw = df.drop(columns=['Fare', 'Survived'])

    # 2. Encode categorical columns
    # Pclass is left as numeric/ordinal.
    # Sex, Embarked -> one-hot encoding
    print("Encoding categorical columns...")
    X = pd.get_dummies(X_raw, columns=['Sex', 'Embarked'], drop_first=True)

    # 3. Train-test split
    print("Splitting dataset...")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # 4. Scaling (prevent data leakage by fitting on train only)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for easier inspection
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 5. Regression (Predicting Fare)
    print("\n--- Regression ---")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_reg_train)
    y_pred_reg = lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_reg_test, y_pred_reg)
    r2_lr = r2_score(y_reg_test, y_pred_reg)

    print(f"Linear Regression MSE: {mse_lr:.2f}")
    print(f"Linear Regression R²: {r2_lr:.4f}")

    # Coefficients
    coeffs = pd.Series(lr.coef_, index=X.columns)
    print("\nLinear Regression Coefficients:")
    print(coeffs)
    print("\nTop 3 absolute coefficients:")
    print(coeffs.abs().nlargest(3))

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_reg_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
    r2_ridge = r2_score(y_reg_test, y_pred_ridge)

    print(f"\nRidge Regression MSE: {mse_ridge:.2f}")
    print(f"Ridge Regression R²: {r2_ridge:.4f}")

    # 6. Classification (Predicting Survival)
    print("\n--- Classification ---")
    print("Class distribution before dealing with imbalance:")
    print(y_clf_train.value_counts())
    
    # We use class_weight='balanced' to handle class imbalance
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_reg.fit(X_train_scaled, y_clf_train)
    y_pred_clf = log_reg.predict(X_test_scaled)
    y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_clf_test, y_pred_clf))
    print("\nClassification Report:")
    print(classification_report(y_clf_test, y_pred_clf))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_clf_test, y_proba)
    auc = roc_auc_score(y_clf_test, y_proba)
    print(f"AUC: {auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = os.path.join(workspace_root, 'capstone project - 2', 'roc_curve.png')
    plt.savefig(roc_path)
    print(f"Saved ROC curve plot to '{roc_path}'")

    # 7. Decision-threshold sensitivity
    print("\nDecision-threshold sensitivity:")
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (y_proba >= threshold).astype(int)
        p = precision_score(y_clf_test, preds)
        r = recall_score(y_clf_test, preds)
        f1 = f1_score(y_clf_test, preds)
        print(f"{threshold:10.2f} | {p:10.4f} | {r:10.4f} | {f1:10.4f}")

    # 8. Regularization experiment
    print("\nRegularization experiment (C=0.01):")
    log_reg_C01 = LogisticRegression(max_iter=1000, C=0.01, class_weight='balanced')
    log_reg_C01.fit(X_train_scaled, y_clf_train)
    y_pred_clf_C01 = log_reg_C01.predict(X_test_scaled)
    y_proba_C01 = log_reg_C01.predict_proba(X_test_scaled)[:, 1]

    p_C01 = precision_score(y_clf_test, y_pred_clf_C01)
    r_C01 = recall_score(y_clf_test, y_pred_clf_C01)
    auc_C01 = roc_auc_score(y_clf_test, y_proba_C01)
    print(f"C=0.01 Model -> Precision: {p_C01:.4f}, Recall: {r_C01:.4f}, AUC: {auc_C01:.4f}")

    # 9. Bootstrap confidence interval for AUC difference
    print("\nRunning Bootstrap for AUC Difference...")
    n_iterations = 500
    auc_diffs = []
    y_clf_test_arr = y_clf_test.values

    np.random.seed(42) # For reproducibility
    for i in range(n_iterations):
        indices = np.random.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
        y_true_boot = y_clf_test_arr[indices]
        
        # Check if both classes are present in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        y_proba_boot_C1 = y_proba[indices]
        y_proba_boot_C01 = y_proba_C01[indices]
        
        auc_C1 = roc_auc_score(y_true_boot, y_proba_boot_C1)
        auc_C01_boot = roc_auc_score(y_true_boot, y_proba_boot_C01)
        
        auc_diffs.append(auc_C1 - auc_C01_boot)

    mean_diff = np.mean(auc_diffs)
    ci_lower = np.percentile(auc_diffs, 2.5)
    ci_upper = np.percentile(auc_diffs, 97.5)

    print(f"Bootstrap AUC Difference (C=1.0 - C=0.01): Mean = {mean_diff:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("Done!")

if __name__ == "__main__":
    main()
