"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : main.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""

from data_quality_checks_and_EDA import quality_checks_and_eda
from PD_model import pd_model
from LGD_and_Exposure_at_Default import lgd_and_exposure
from ECL_Expected_Credit_Loss import ecl_and_visualization
from stress_testing import stress_testing

def main():
    df = quality_checks_and_eda()
    X, _, cal_pd, num_features, cat_features = pd_model(df)
    df_lgd, true_defaults = lgd_and_exposure(df, X, cal_pd, num_features, cat_features)
    df_lgd, discount_factor = ecl_and_visualization(df_lgd, true_defaults)
    stress_testing(df_lgd, discount_factor)

if __name__ == "__main__":
    main()

