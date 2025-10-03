"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : stress_testing.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""

def stress_testing(df_lgd, discount_factor):

    import numpy as np
    np.set_printoptions(precision=3)

    # ===========[[ Stress testing predictions ]]============{{{
    print(f"")
    print(68*"=")
    print(f"==={15*'='}[[ Stress testing predictions ]]{15*'='}===\n")
    # ==========================================================

    """# {{{

    Stress-test portfolio under adverse conditions:
        - PD +50%
        - LGD +20%
        - Both combined
        - Severe scenario (PD×2, LGD×1.5)

    Reporting changes in portfolio ECL vs base case.

    """# }}}

    # already computed
    base_ecl = df_lgd["ECL_12m"].sum().round(2)
    print("Base case ECL (12m):", base_ecl)

    # stress scenarios
    stress_scenarios = {
        "PD +50%": {"PD": 1.5, "LGD": 1.0},
        "LGD +20%": {"PD": 1.0, "LGD": 1.2},
        "PD +50% & LGD +20%": {"PD": 1.5, "LGD": 1.2},
        "Severe stress (PD×2, LGD×1.5)": {"PD": 2.0, "LGD": 1.5}
    }

    # Apply scenarios
    results = {}
    for name, factors in stress_scenarios.items():
        df_stress = df_lgd.copy()
        df_stress["PD_stress"] = (df_stress["PD"] * factors["PD"]).clip(upper=1.0)
        df_stress["LGD_stress"] = (df_stress["LGD_exp"] * factors["LGD"]).clip(upper=1.0)
        df_stress["ECL_stress"] = (
            df_stress["PD_stress"] * df_stress["LGD_stress"] * df_stress["EAD"] * discount_factor
        )
        results[name] = df_stress["ECL_stress"].sum().round(2)

    # results
    print("\nStress Test Results")
    for scenario, ecl in results.items():
        print(f"    {scenario:30} : {ecl}  (vs base {base_ecl}, change {((ecl/base_ecl)-1)*100:.1f}%)")

    # }}}

    return results

if __name__ == "__main__":
    from data_quality_checks_and_EDA import quality_checks_and_eda
    from PD_model import pd_model
    from LGD_and_Exposure_at_Default import lgd_and_exposure
    from ECL_Expected_Credit_Loss import ecl_and_visualization

    df = quality_checks_and_eda()
    X, _, cal_pd, num_features, cat_features = pd_model(df)
    df_lgd, true_defaults = lgd_and_exposure(df, X, cal_pd, num_features, cat_features)
    df_lgd, discount_factor = ecl_and_visualization(df_lgd, true_defaults)
    stress_testing(df_lgd, discount_factor)
