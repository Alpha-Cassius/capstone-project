import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    print("--- Part 3: Advanced Modeling & Pipeline ---\n")
    
    # 1. Data Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))
    data_path = os.path.join(workspace_root, 'capstone project - 1', 'data', 'processed', 'cleaned_data.csv')
    
    df = pd.read_csv(data_path)
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna().reset_index(drop=True)
    
    y_clf = df['Survived']
    X_raw = df.drop(columns=['Survived'])
    X = pd.get_dummies(X_raw, columns=['Sex', 'Embarked'], drop_first=True)
    
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 1. Decision Tree Baseline
    print("\n--- 1. Decision Tree Baseline ---")
    dt_base = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_base.fit(X_train_scaled, y_clf_train)
    print(f"Unconstrained DT - Train Accuracy: {accuracy_score(y_clf_train, dt_base.predict(X_train_scaled)):.4f}")
    print(f"Unconstrained DT - Test Accuracy: {accuracy_score(y_clf_test, dt_base.predict(X_test_scaled)):.4f}")

    # 2. Controlled Decision Tree
    print("\n--- 2. Controlled Decision Tree ---")
    dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    dt_ctrl.fit(X_train_scaled, y_clf_train)
    print(f"Controlled DT - Train Accuracy: {accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled)):.4f}")
    print(f"Controlled DT - Test Accuracy: {accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled)):.4f}")

    # 3. Gini vs Entropy Comparison
    print("\n--- 3. Gini vs Entropy ---")
    dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    dt_gini.fit(X_train_scaled, y_clf_train)
    
    dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
    dt_entropy.fit(X_train_scaled, y_clf_train)
    
    print(f"Gini DT Test Accuracy: {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)):.4f}")
    print(f"Entropy DT Test Accuracy: {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)):.4f}")

    # 4. Random Forest
    print("\n--- 4. Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_clf_train)
    rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
    rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
    rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
    
    print(f"RF Train Accuracy: {rf_train_acc:.4f}, Test Accuracy: {rf_test_acc:.4f}, AUC: {rf_auc:.4f}")
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nTop 5 RF Features by Importance:")
    print(importances.nlargest(5))

    # 4a. Gradient Boosting
    print("\n--- 4a. Gradient Boosting ---")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X_train_scaled, y_clf_train)
    gbc_train_acc = accuracy_score(y_clf_train, gbc.predict(X_train_scaled))
    gbc_test_acc = accuracy_score(y_clf_test, gbc.predict(X_test_scaled))
    gbc_auc = roc_auc_score(y_clf_test, gbc.predict_proba(X_test_scaled)[:, 1])
    print(f"GBC Train Accuracy: {gbc_train_acc:.4f}, Test Accuracy: {gbc_test_acc:.4f}, AUC: {gbc_auc:.4f}")

    # 4b. Feature Ablation Study
    print("\n--- 4b. Feature Ablation Study ---")
    lowest_5_features = importances.nsmallest(5).index.tolist()
    print("5 Lowest Importance Features:", lowest_5_features)
    
    X_train_ablated = X_train_scaled.drop(columns=lowest_5_features)
    X_test_ablated = X_test_scaled.drop(columns=lowest_5_features)
    
    rf_ablated = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_ablated.fit(X_train_ablated, y_clf_train)
    rf_ablated_auc = roc_auc_score(y_clf_test, rf_ablated.predict_proba(X_test_ablated)[:, 1])
    
    print(f"Full RF AUC: {rf_auc:.4f}")
    print(f"Ablated RF AUC: {rf_ablated_auc:.4f}")

    # 5. Cross-validated Comparison
    print("\n--- 5. Cross-validated Comparison ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    models = {
        'Logistic Regression': log_reg,
        'Controlled Decision Tree': dt_ctrl,
        'Random Forest': rf,
        'Gradient Boosting': gbc
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring='roc_auc')
        print(f"{name} -> 5-fold AUC Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

    # 6. Hyperparameter Tuning with GridSearchCV
    print("\n--- 6. Hyperparameter Tuning ---")
    pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        RandomForestClassifier(random_state=42)
    )
    
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [5, 10, None],
        'randomforestclassifier__min_samples_leaf': [1, 5]
    }
    
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_clf_train) # Fit on unscaled data, pipeline handles it
    
    print(f"Best Params: {grid.best_params_}")
    print(f"Best Score (AUC): {grid.best_score_:.4f}")

    # 7. Manual Learning Curve
    print("\n--- 7. Manual Learning Curve ---")
    best_pipeline = grid.best_estimator_
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"{'Training fraction':>17} | {'Training AUC':>12} | {'Test AUC':>10}")
    for f in fractions:
        limit = int(f * len(X_train))
        X_subset = X_train.iloc[:limit]
        y_subset = y_clf_train.iloc[:limit]
        
        best_pipeline.fit(X_subset, y_subset)
        
        train_auc = roc_auc_score(y_subset, best_pipeline.predict_proba(X_subset)[:, 1])
        test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
        
        print(f"{f:17.2f} | {train_auc:12.4f} | {test_auc:10.4f}")

    # 8. Serialize the best model
    print("\n--- 8. Serialization ---")
    model_path = os.path.join(workspace_root, 'capstone project - 3', 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Reload and predict on two dummy rows
    loaded_model = joblib.load(model_path)
    
    dummy_data = pd.DataFrame({
        'Pclass': [1, 3],
        'Age': [30.0, 22.0],
        'SibSp': [0, 1],
        'Parch': [0, 0],
        'Sex_male': [1, 0],
        'Embarked_Q': [0, 1],
        'Embarked_S': [1, 0]
    })
    
    preds = loaded_model.predict(dummy_data)
    print(f"\nPredictions on handcrafted rows:")
    print(dummy_data)
    print(f"Predicted Survived: {preds}")

if __name__ == "__main__":
    main()
