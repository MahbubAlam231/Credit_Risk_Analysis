"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : LGD_and_Exposure_at_Default.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""


def lgd_and_exposure(df, X, cal_pd, num_features, cat_features):

    import numpy as np
    np.set_printoptions(precision=3)

    # =======================[[ LGD ]]======================={{{
    print(f"")
    print(68*"=")
    print(f"==={27*'='}[[ LGD ]]{27*'='}==\n")
    # ==========================================================

    """# {{{

    The german credit dataset lacks information on recoveries for defaulted loans,
    so Loss Given Default (LGD) cannot be observed directly.
    We **simulate LGD using a rule-based approach with random noise**
    based on borrowers property, account status, savings, and purpose.

    Later we will **train a RandomForest model to estimate LGD from
    historical data** (treating part our simulation as historical data).
    Predictions are floored at 10% in line with regulatory requirements [CRR Article 164(4)](https://www.eba.europa.eu/single-rule-book-qa/qna/view/publicId/2017_3554).

    """# }}}

    df_lgd = df.copy()

    def lgd_simulation(row):
        # setting base lgd
        lgd = 0.55

        prop = row.get("Property").lower()
        check = row.get("Status_of_existing_checking_account").lower()
        savings = row.get("Savings_account_bonds").lower()
        purpose = row.get("Purpose").lower()

        # Collateral
        if "real estate" in prop or "building" in prop:
            lgd -= 0.20
        elif "car" in prop:
            lgd -= 0.10
        elif "unknown" in prop:
            lgd += 0.05

        # Liquidity buffers
        if ">= 200" in check:
            lgd -= 0.05
        if ">= 1000" in savings or "rich" in savings:
            lgd -= 0.05

        # Purposes that are typically less collateralized
        if purpose in ["education", "retraining", "others"]:
            lgd += 0.05

        return float(np.clip(lgd, 0.05, 0.95))

    df_lgd["LGD_prior"] = df_lgd.apply(lgd_simulation, axis=1)

    # Adding a randon noise to our simulation
    rng = np.random.default_rng(42)
    df_lgd["LGD_obs"] = np.where(
        df_lgd["Risk"] == 1,
        np.clip(df_lgd["LGD_prior"] + rng.normal(0, 0.01, size=len(df_lgd)), 0.01, 0.99),
        0.0
    )

    # ===============[[ Output title like this ]]===============
    print(f"")
    print(68*"=")
    print(f"==={16*'='}[[ Simulate true defaulters ]]{16*'='}===\n")
    # ==========================================================

    """# {{{
    Summary:
        - Sample 90% of bad credits as "true defaulters", using PD-weighted probabilities.
        - Assume that only those bad credits truly defaulted.
        - Use this as "historical training data" for LGD modeling.
    """# }}}

    df_lgd["PD"] = cal_pd.predict_proba(X)[:,1]

    bad_index = np.where(df_lgd["Risk"] == 1)[0]
    weights = df_lgd.loc[bad_index, "PD"].to_numpy()
    weights = weights / weights.sum()

    rng = np.random.default_rng(43)
    true_defaults = rng.choice(bad_index, size=int(0.9 * len(bad_index)), replace=False, p=weights)

    df_lgd["true_default"] = 0
    df_lgd.loc[true_defaults, "true_default"] = 1

    # ===============[[ Output title like this ]]===============
    print(f"")
    print(68*"=")
    print(f"==={15*'='}[[ Random forest model for LGD ]]{15*'='}==\n")
    # ==========================================================

    """# {{{
    Summary:
        - Train Random Forest regressor on simulated LGD data.
        - Predict expected LGD (LGD_exp) for entire portfolio.
        - Expected LGD predictions are floored at 10% to align with CRR Article 164(4).

    """# }}}

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    X_lgd = df_lgd.loc[true_defaults].drop(columns=["Risk", "PD", "true_default", "LGD_prior", "LGD_obs"]).copy()
    y_lgd = df_lgd.loc[true_defaults, "LGD_obs"]
    # print(y_lgd.head())

    lgd_pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features),
        ]
    )

    lgd_model = Pipeline([
        ("pre", lgd_pre),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=40))
    ])

    lgd_model.fit(X_lgd, y_lgd)

    df_lgd["LGD_exp"] = np.clip(
        lgd_model.predict(df_lgd.drop(columns=["Risk", "true_default", "LGD_prior", "LGD_obs"])),
        0.10, 0.99
    )

    # }}}

    # ===============[[ Exposure at Default ]]==============={{{
    print(f"")
    print(68*"=")
    print(f"==={19*'='}[[ Exposure at Default ]]{19*'='}==\n")
    # ==========================================================

    """# {{{

    Simulating EAD since the dataset lacks this column.

    Formula (Basel/IFRS 9 style):
    ```
        EAD = Balance + CCF * Undrawn

        Balance = Amount withdrawn - Amount repaid

        CCF = Credit Conversion Factor = Percentage defaulters withdraw just before defaulting
    ```

    In practice, banks use observed balances and limits.
    Assumptions:
    - Base utilization: random 50–95% of the credit amount
    - Risk adjustment: riskier borrowers (high PD) repay less
    - CCF fixed at 75% (typical Basel retail assumption)

    """# }}}

    rng = np.random.default_rng(50)

    base_frac = np.clip(rng.normal(0.65, 0.1, size=len(df_lgd)), 0.3, 0.8)

    risk_adj = df_lgd["PD"] * 0.2
    balance_frac = np.clip(base_frac + risk_adj, 0.5, 0.95)

    df_lgd["Withdrawn"] = df_lgd["Credit_amount"] * balance_frac
    df_lgd["Undrawn"] = df_lgd["Credit_amount"] - df_lgd["Withdrawn"]

    repay_frac = (1 - df_lgd["PD"]) * 0.8
    df_lgd["Repaid"] = df_lgd["Withdrawn"] * repay_frac
    df_lgd["Balance"] = df_lgd["Withdrawn"] - df_lgd["Repaid"]

    CCF = 0.8

    df_lgd["EAD"] = df_lgd["Balance"] + CCF * df_lgd["Undrawn"]

    # }}}

    return df_lgd, true_defaults

if __name__ == "__main__":
    from data_quality_checks_and_EDA import quality_checks_and_eda
    from PD_model import pd_model

    df = quality_checks_and_eda()
    X, _, cal_pd, num_features, cat_features = pd_model(df)
    lgd_and_exposure(df, X, cal_pd, num_features, cat_features)
