"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : ECL_Expected_Credit_Loss.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""


def ecl_and_visualization(df_lgd, true_defaults):

    import numpy as np
    np.set_printoptions(precision=3)
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from utils import wrap_labels_

    # ==============[[ Expected Credit Loss ]]==============={{{
    print(f"")
    print(68*"=")
    print(f"==={18*'='}[[ Expected Credit Loss ]]{18*'='}===\n")
    # ==========================================================

    """# {{{

    This block computes 12-month ECL.
      - Formula:
          ```
          ECL = PD * LGD * EAD * discount factor
          ```
    We report portfolio-level stats and segment summaries:
      - By Purpose
      - By Property * EAD buckets (qcut)

    """# }}}

    # EIR, one year discount, assumed flat 5% for demonstration
    annual_rate = 0.05
    discount_factor = 1/(1+annual_rate)

    df_lgd["ECL_12m"] = df_lgd["PD"] * df_lgd["LGD_exp"] * df_lgd["EAD"] * discount_factor

    print(df_lgd.loc[true_defaults, ["Credit_amount", "PD", "LGD_exp", "EAD", "ECL_12m"]].head())

    # Portfolio summaries
    print("\n=== Portfolio summary ===")
    print("Total EAD:", df_lgd["EAD"].sum().round(2))
    print("Total 12m ECL:", df_lgd["ECL_12m"].sum().round(2))
    print("Average PD:", df_lgd["PD"].mean().round(3))
    print("Average LGD (expected):", df_lgd["LGD_exp"].mean().round(3))
    print("ECL / EAD (portfolio charge):", (df_lgd["ECL_12m"].sum() / df_lgd["EAD"].sum()).round(4))

    # segment views; purpose and ECL_12m
    purpose_ecl = df_lgd.groupby("Purpose")["ECL_12m"].sum()
    purpose_ead = df_lgd.groupby("Purpose")["EAD"].sum()
    print("\n=== Segment summary (by Purpose) ===")
    print(purpose_ecl)
    print(f"")
    print("ECL density by Purpose:", (purpose_ecl / purpose_ead).round(3))

    # Segment views ('Property' and credit size bucket)
    EAD_bins = pd.qcut(df_lgd["EAD"], q=4, duplicates="drop")

    seg = (
        df_lgd.groupby(["Property", EAD_bins], observed=True).agg(
            n_accounts = ("PD", "size"),
            avg_PD = ("PD", lambda x : round(x.mean(), 3)),
            avg_LGD = ("LGD_exp", lambda x : round(x.mean(), 3)),
            total_EAD = ("EAD", lambda x : round(x.sum(), 2)),
            total_ECL_12m = ("ECL_12m", lambda x : round(x.sum(), 2))
        )
        .reset_index()
        .rename(columns={"EAD": "EAD_bucket"})
    )

    seg["EAD_bucket"] = seg["EAD_bucket"].apply(
        lambda interval: f"({round(interval.left)}, {round(interval.right)}]"
    )
    seg["ECL_density"] = (seg["total_ECL_12m"] / seg["total_EAD"]).round(4)

    seg.to_csv('segment_view_risk.csv', index=False)

    print("\n=== Segment summary (Property x EAD quartiles) ===")
    print(seg.head(12))

    # }}}

    # ===========[[ Visualizing the predictions ]]==========={{{
    print(f"")
    print(68*"=")
    print(f"==={15*'='}[[ Visualizing the predictions ]]{15*'='}==\n")
    # ==========================================================

    """# {{{

    Visualize results with:
        - ECL by Purpose (barplot)
        - ECL density by EAD bucket
        - Total EAD vs Total ECL by Property
        - ECL density heatmap (Property * EAD bucket)

    """# }}}

    pur_summary = pd.DataFrame(
        purpose_ecl.sort_values(ascending=False).reset_index(),
        columns = np.array(["Purpose", "ECL_12m"])
    )

    plt.figure(num="ECL by Loan Purpose", figsize=(7,4))
    sns.barplot(data=pur_summary, x="ECL_12m", y="Purpose", hue="Purpose", palette="Reds_r")
    plt.xlabel("Total 12m ECL")
    plt.ylabel("Purpose")
    plt.title("ECL by Loan Purpose")
    plt.tight_layout()
    plt.savefig(f'ECL_by_Loan_Purpose.jpg')
    plt.show()

    _, axes = plt.subplots(1, 2, figsize=(12, 6), num="risk_vs_acc_status_credit_amount_and_age")
    # ECL density by EAD bucket
    ax = sns.barplot(
        data=seg,
        x="EAD_bucket",
        y="ECL_density",
        estimator=sum,
        errorbar=None,
        ax=axes[0]
    )

    wrap_labels_(ax, width=18, rotation=30, ha="at_tick", pad=10)
    ax.set_title("ECL Density by EAD Bucket")
    ax.set_ylabel("ECL Density (ECL/EAD)")
    ax.set_xlabel("EAD Bucket")

    # Stacked bar chart total EAD and total ECL by Property
    prop_summary = seg.sort_values("total_EAD", ascending=False)

    axes[1].bar(prop_summary["Property"], prop_summary["total_EAD"], label="Total EAD")
    axes[1].bar(prop_summary["Property"], prop_summary["total_ECL_12m"], label="Total ECL (12m)")
    axes[1].set_title("Total EAD and ECL by Property")
    axes[1].set_ylabel("Amount")
    axes[1].set_xlabel("Property")
    wrap_labels_(axes[1], width=18, rotation=30, ha="at_tick", pad=10)
    axes[1].legend()

    for ax in axes:
        ax.xaxis.set_label_coords(0.5, -0.35)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.25)
    plt.savefig('ECL_density_total_EAD_EAD_bucket_Property.jpg')
    plt.show()

    # ECL density heatmap (Property * EAD bucket)
    heatmap_data = seg.pivot_table(index="Property", columns="EAD_bucket",
                                   values="ECL_density", aggfunc="mean", observed=True)

    plt.figure(num="ECL Density Heatmap (Property * EAD bucket)", figsize=(9,5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="Reds")
    plt.title("ECL Density Heatmap (Property * EAD bucket)")
    plt.ylabel("Property")
    plt.xlabel("EAD Bucket")
    plt.gca().xaxis.set_label_coords(0.5, -0.15)
    plt.tight_layout()
    plt.savefig(f'ECL_density_heatmap.jpg')
    plt.show()

    # }}}

    return df_lgd, discount_factor

if __name__ == "__main__":
    from data_quality_checks_and_EDA import quality_checks_and_eda
    from PD_model import pd_model
    from LGD_and_Exposure_at_Default import lgd_and_exposure

    df = quality_checks_and_eda()
    X, _, cal_pd, num_features, cat_features = pd_model(df)
    df_lgd, true_defaults = lgd_and_exposure(df, X, cal_pd, num_features, cat_features)
    ecl_and_visualization(df_lgd, true_defaults)
