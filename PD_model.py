"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : PD_model.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""

from data_quality_checks_and_EDA import quality_checks_and_eda


def pd_model(df):

    import numpy as np
    np.set_printoptions(precision=3)

    # ====================[[ PD model ]]====================={{{
    print(f"")
    print(68*"=")
    print(f"==={24*'='}[[ PD model ]]{24*'='}===\n")
    # ==========================================================

    """# {{{

    This is the Probability of Default (PD) modeling step (Basel/IFRS 9).
        - Train Logistic Regression model to predict PD.
        - Use preprocessing pipeline with scaling + one-hot encoding.
        - Tune hyperparameters via GridSearchCV (refit on ROC-AUC).
        - Calibrate final model using isotonic regression to align predicted PDs with observed default rates (CalibratedClassifierCV).

    """# }}}

    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (roc_auc_score, brier_score_loss,
                                 make_scorer, average_precision_score, roc_curve,
                                 confusion_matrix, accuracy_score, recall_score)

    X = df.drop(columns=['Risk']).copy()
    y = df['Risk']

    num_features = X.columns[X.dtypes.apply(lambda dt : np.issubdtype(dt, np.number))].tolist()
    cat_features = [col for col in X.columns if col not in num_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y)

    pre = ColumnTransformer(
        transformers = [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features)
        ])

    pd_pipe = Pipeline([
        ("pre", pre),
        ("logreg", LogisticRegression(max_iter = 1000, solver = "liblinear", random_state = 1))
    ])

    hparams = {
        'logreg__penalty': ['l1', 'l2'],
        'logreg__C': np.logspace(-3, 3, 13),
        'logreg__class_weight': [None, 'balanced'],
        'logreg__fit_intercept': [True, False],
        'logreg__tol': [1e-4, 1e-5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 2)

    scorers = {
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
        "brier": make_scorer(brier_score_loss, response_method="predict_proba", greater_is_better=False)
    }

    grid = GridSearchCV(
        pd_pipe,
        hparams,
        scoring = scorers,
        cv = cv,
        n_jobs = 4,
        refit = "roc_auc"
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Calibrate best model using isotonic regression, this ensures PDs align with observed default rates
    cal_pd = CalibratedClassifierCV(estimator = best_model, method = "isotonic", cv = 5)
    cal_pd.fit(X_train, y_train)

    # ===============[[ Output title like this ]]===============
    print(f"")
    print(68*"=")
    print(f"==={22*'='}[[ Model report ]]{22*'='}===\n")
    # ==========================================================

    """# {{{

    Generating validation report for the PD model. Discrimination (ROC-AUC, Gini) and calibration (Brier score) are the primary validation metrics.

          - ROC-AUC and Gini coefficient (discrimination power)
          - Brier score (calibration quality)
          - Average precision score
          - KS statistic (max separation between good/bad)

        Threshold-based metrics at KS-optimal threshold:
          - KS-optimal threshold
          - Metrics at that threshold: Accuracy, Recall, Specificity, Confusion matrix

    """# }}}

    proba_test = cal_pd.predict_proba(X_test)[:,1]
    print(f"Test AUC: {roc_auc_score(y_test, proba_test):.3f}")
    gini = 2*roc_auc_score(y_test, proba_test) - 1
    print(f"Gini coefficient: {gini:.3f}")
    print(f"Brier score: {brier_score_loss(y_test, proba_test):.4f}")
    print(f"Average precision score: {average_precision_score(y_test, proba_test):.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    ks_index = (tpr - fpr).argmax()
    ks_stat = tpr[ks_index] - fpr[ks_index]
    print(f"KS Statistic: {ks_stat:.3f}")

    print(f"")
    y_pred_ks = (proba_test >= thresholds[ks_index]).astype(int)
    print(f"Optimal KS threshold: {thresholds[ks_index]:.3f}")
    print("Diagnostic metrics at KS threshold:")
    print(f"    Accuracy: {accuracy_score(y_test, y_pred_ks):.3f}")
    print(f"    Recall: {recall_score(y_test, y_pred_ks):.3f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ks).ravel()
    specificity = tn / (tn + fp)
    print(f"    Specificity (good capture rate): {specificity:.3f}")
    print(f"    Confusion Matrix:\n {confusion_matrix(y_test, y_pred_ks)}")

    # ===============[[ Output title like this ]]===============
    print(f"")
    print(68*"=")
    print(f"==={16*'='}[[ Save the model for later ]]{16*'='}===\n")
    # ==========================================================

    import joblib

    joblib.dump(cal_pd, "pd_model_calibrated.pkl")

    # Load model
    # cal_pd = joblib.load("pd_model_calibrated.pkl")

    # }}}

    return X, y, cal_pd, num_features, cat_features

if __name__ == "__main__":
    from data_quality_checks_and_EDA import quality_checks_and_eda
    df = quality_checks_and_eda()
    pd_model(df)
