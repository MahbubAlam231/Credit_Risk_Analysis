"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : data_quality_checks_and_EDA.py
Created     : 2025-08
Description : Credit Risk Analysis using the German Credit dataset. # {{{

# }}}
"""

def quality_checks_and_eda():

    import numpy as np
    np.set_printoptions(precision=3)
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns
    import json

    # ===============[[ Data quality checks ]]==============={{{
    print(f"")
    print(68*"=")
    print(f"==={19*'='}[[ Data quality checks ]]{19*'='}==\n")
    # ==========================================================

    """# {{{

    Summary:

    - Inspect schema and datatypes
    - Check for missing values
    - Remove duplicates
    - Count unique values per column
    - Check class balance of the target (Risk: good/bad)

    """# }}}

    df = pd.read_csv('full_german_credit.csv')
    # print(df.head())

    print(f"")
    print(df.info())

    # Check missing data
    missing_data = df.isna().any()
    print(f"Missing data? - {'Yes' if missing_data.any() else 'No'}")

    # Drop perfect duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # check unique values
    nunique = df.nunique().to_dict()
    print(f"")
    print(f"Unique values per column:\n\n{json.dumps(nunique, indent=4)}")

    # Zero-variance check
    zero_var = np.array([k for k in nunique if nunique[k] <= 1])
    print(f"")
    print(f"No. of columns with no variability (only one unique value): {len(zero_var)}")

    print(f"")
    # print(df['Risk'].head())
    vc = df['Risk'].value_counts()
    vcp = df['Risk'].value_counts(normalize=True)

    class_balance = pd.DataFrame({'count' : vc, 'proportion' : vcp})

    print(f"")
    print(f"Class balance of the target (Risk):\n\n{class_balance}")

    # ===============[[ Output title like this ]]===============
    print(f"")
    print(68*"=")
    print(f"==={22*'='}[[ Encoding Risk ]]{22*'='}==\n")
    # ==========================================================

    """# {{{

    Encoding target variable Risk into numeric (1=bad, 0=good).
    This prepares the dataset for supervised learning models.

    """# }}}

    df["Risk"] = (df["Risk"] == "bad").astype(int)

    # }}}

    # ===[[ EDA with Account status, Loan amount and Age ]]===={{{
    print(f"")
    print(68*"=")
    print(f"==={6*'='}[[ EDA with Account status, Loan amount and Age ]]{6*'='}===\n")
    # ==========================================================

    """# {{{

    Exploring how credit risk correlates with key features such as:
          - Account status (barplot of default rates)
          - Loan amount (boxplots by Risk)
          - Age (histograms + median lines by Risk)

    """# }}}

    from utils import wrap_labels_

    # def wrap_labels_(ax, width=15, rotation=0, ha="right", pad=5):# {{{
    #     """Wrap long labels on x-axis"""
    #     ticks = ax.get_xticks()
    #     labels = [label.get_text() for label in ax.get_xticklabels()]

    #     # Wrap long labels
    #     wrapped_labels = ["\n".join(textwrap.wrap(l, width=width)) for l in labels]

    #     # Set ticks + labels explicitly
    #     ax.set_xticks(ticks)
    #     if ha == "at_tick":
    #         ax.set_xticklabels(wrapped_labels, rotation=rotation)
    #         for label in ax.get_xticklabels():
    #             label.set_x(label.get_position()[0])
    #     else:
    #         ax.set_xticklabels(wrapped_labels, rotation=rotation, ha=ha)

    #     # Adjust padding
    #     ax.tick_params(axis="x", pad=pad)

    # # }}}

    _, axes = plt.subplots(1, 3, figsize=(18, 6), num="risk_vs_acc_status_credit_amount_and_age")
    # risk vs account status
    ax = sns.barplot(
        x="Status_of_existing_checking_account",
        y="Risk",
        data=df,
        errorbar=None,
        estimator=np.mean,
        ax=axes[0]
    )

    # adding percentage on top of the bars
    for p in ax.patches:
        value = p.get_height()
        ax.annotate(f"{value:.0%}",
                    (p.get_x() + p.get_width()/2, value),
                    ha="center", va="bottom", fontsize=10)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    wrap_labels_(ax, width=18, rotation=30, ha="at_tick", pad=10)
    ax.set_title("Default Rate by Checking Account Status")
    ax.set_ylabel("Default Rate")

    # risk vs credit amount
    ax = sns.boxplot(
        x="Risk",
        y="Credit_amount",
        data=df,
        # showfliers=False,
        hue="Risk",
        dodge=False,
        palette="Set2",
        ax=axes[1]
    )

    # Add median annotations
    medians = df.groupby("Risk")["Credit_amount"].median()
    for i, median in enumerate(medians):
        ax.text(i, median, f"{median:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
    ax.set_ylim(0, 10000)
    ax.set_title("Loan Amount Distribution by Credit Risk")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Credit Amount")

    # risk vs age
    ax = sns.histplot(
        data=df,
        x="Age_in_years",
        hue="Risk",
        bins=10,
        multiple="dodge",
        stat="percent",
        common_norm=False,
        palette="Set2",
        ax=axes[2]
    )

    # Add median lines for each group
    medians = df.groupby("Risk")["Age_in_years"].median()
    colors = {0: "green", 1: "red"}

    for risk_val, median in medians.items():
        ax.axvline(median, color=colors[risk_val], linestyle="--", linewidth=1.5)
        ax.text(median, ax.get_ylim()[1]*0.9, f"Median {median:.0f}",
                rotation=90, ha="left", va="top", fontsize=10)

    ax.set_title("Age Distribution by Credit Risk")
    ax.set_xlabel("Age_in_years")
    ax.set_ylabel("Share of Customers (%)")

    for ax in axes:
        ax.xaxis.set_label_coords(0.5, -0.45)

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(wspace=0.25)
    plt.savefig('risk_vs_acc_status_credit_amount_and_age.jpg')
    plt.show()

    # }}}

    return df

if __name__ == "__main__":
    quality_checks_and_eda()
