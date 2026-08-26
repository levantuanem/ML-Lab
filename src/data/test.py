import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIG
# ============================================================

DATA_PATH = r"D:\AI_Projects\ML-Lab\data\raw\Bank_churn.csv"

TARGET = "Exited"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

data = pd.read_csv(DATA_PATH)

print(f"Shape : {data.shape}")
print(f"Rows  : {data.shape[0]:,}")
print(f"Cols  : {data.shape[1]}")


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("2. BASIC DATA INFORMATION")
print("=" * 70)

print("\nColumns:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nDataFrame info:")
data.info()


# ============================================================
# 4. COLUMN AUDIT
# ============================================================

print("\n" + "=" * 70)
print("3. COLUMN AUDIT")
print("=" * 70)

column_audit = pd.DataFrame({
    "Column": data.columns,
    "Data_Type": data.dtypes.astype(str),
    "Missing_Count": data.isna().sum(),
    "Missing_%": (
        data.isna().mean() * 100
    ).round(2),
    "Unique_Count": data.nunique(
        dropna=True
    )
})

print(
    column_audit.to_string(
        index=False
    )
)


# ============================================================
# 5. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET ANALYSIS")
print("=" * 70)

if TARGET not in data.columns:

    print(
        f"WARNING: Target '{TARGET}' "
        "does not exist!"
    )

else:

    print(f"\nTarget: {TARGET}")

    print("\nUnique values:")
    print(
        data[TARGET]
        .unique()
    )

    print("\nValue counts:")
    print(
        data[TARGET]
        .value_counts(
            dropna=False
        )
    )

    print("\nTarget distribution (%):")

    target_distribution = (
        data[TARGET]
        .value_counts(
            normalize=True,
            dropna=False
        )
        * 100
    ).round(2)

    print(target_distribution)


# ============================================================
# 6. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING VALUE ANALYSIS")
print("=" * 70)

missing = pd.DataFrame({
    "Column": data.columns,
    "Missing_Count": data.isna().sum(),
    "Missing_%": (
        data.isna().mean() * 100
    ).round(2)
})

missing = (
    missing[
        missing["Missing_Count"] > 0
    ]
    .sort_values(
        "Missing_Count",
        ascending=False
    )
)

if missing.empty:

    print("No missing values found.")

else:

    print(
        missing.to_string(
            index=False
        )
    )


# ============================================================
# 7. DUPLICATE ROW ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. DUPLICATE ROW ANALYSIS")
print("=" * 70)

duplicate_rows = data.duplicated().sum()

print(
    f"Duplicate rows: "
    f"{duplicate_rows:,}"
)

if duplicate_rows == 0:

    print(
        "OK - No duplicate rows."
    )

else:

    print(
        "WARNING - Duplicate rows found!"
    )


# ============================================================
# 8. IDENTIFIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. IDENTIFIER ANALYSIS")
print("=" * 70)

identifier_columns = [
    "id",
    "CustomerId",
    "Surname"
]

for column in identifier_columns:

    if column not in data.columns:

        print(
            f"\n{column}: NOT FOUND"
        )

        continue

    print(f"\n--- {column} ---")

    print(
        "Data type:",
        data[column].dtype
    )

    print(
        "Missing:",
        data[column].isna().sum()
    )

    print(
        "Unique:",
        data[column].nunique(
            dropna=True
        )
    )

    unique_ratio = (
        data[column]
        .nunique(dropna=True)
        / len(data)
        * 100
    )

    print(
        f"Unique ratio: "
        f"{unique_ratio:.2f}%"
    )

    duplicate_count = (
        data[column]
        .duplicated()
        .sum()
    )

    print(
        "Duplicated values:",
        duplicate_count
    )


# ============================================================
# 9. DUPLICATE CUSTOMER ID
# ============================================================

print("\n" + "=" * 70)
print("8. CUSTOMER ID DUPLICATE CHECK")
print("=" * 70)

if "CustomerId" in data.columns:

    duplicate_customer = (
        data["CustomerId"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate CustomerId:",
        duplicate_customer
    )

    if duplicate_customer == 0:

        print(
            "OK - No duplicated CustomerId."
        )

    else:

        print(
            "WARNING - "
            "Duplicated CustomerId found!"
        )


# ============================================================
# 10. NUMERICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("9. NUMERICAL FEATURES")
print("=" * 70)

numeric_columns = data.select_dtypes(
    include=np.number
).columns.tolist()

print(
    "\nNumerical columns:"
)

print(numeric_columns)

if numeric_columns:

    print("\nDescriptive statistics:")

    print(
        data[numeric_columns]
        .describe()
        .T
        .round(2)
    )


# ============================================================
# 11. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("10. CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(
    "\nCategorical columns:"
)

print(categorical_columns)

for column in categorical_columns:

    print(
        f"\n--- {column} ---"
    )

    print(
        data[column]
        .value_counts(
            dropna=False
        )
        .head(30)
    )


# ============================================================
# 12. EXPECTED BANK CHURN FEATURES
# ============================================================

print("\n" + "=" * 70)
print("11. EXPECTED FEATURE CHECK")
print("=" * 70)

expected_columns = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

for column in expected_columns:

    if column in data.columns:

        print(
            f"[OK]      {column}"
        )

    else:

        print(
            f"[MISSING] {column}"
        )


# ============================================================
# 13. BINARY VALUE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("12. BINARY VALUE VALIDATION")
print("=" * 70)

binary_columns = [
    "HasCrCard",
    "IsActiveMember",
    "Exited"
]

for column in binary_columns:

    if column not in data.columns:
        continue

    print(
        f"\n--- {column} ---"
    )

    values = (
        data[column]
        .dropna()
        .unique()
    )

    print(
        "Unique values:",
        sorted(values)
    )

    invalid_values = [
        value
        for value in values
        if value not in [0, 1]
    ]

    if invalid_values:

        print(
            "WARNING - Invalid values:",
            invalid_values
        )

    else:

        print(
            "OK - Binary values."
        )


# ============================================================
# 14. RANGE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("13. RANGE VALIDATION")
print("=" * 70)


def check_range(
    column,
    min_value=None,
    max_value=None
):

    if column not in data.columns:
        return

    series = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    invalid = pd.Series(
        False,
        index=data.index
    )

    if min_value is not None:

        invalid |= (
            series < min_value
        )

    if max_value is not None:

        invalid |= (
            series > max_value
        )

    count = invalid.sum()

    print(
        f"\n--- {column} ---"
    )

    print(
        "Min:",
        series.min()
    )

    print(
        "Max:",
        series.max()
    )

    print(
        "Invalid count:",
        count
    )

    if count > 0:

        print(
            "Examples:"
        )

        print(
            data.loc[
                invalid,
                column
            ]
            .head(10)
            .tolist()
        )


# Credit Score
check_range(
    "CreditScore",
    min_value=0,
    max_value=1000
)

# Age
check_range(
    "Age",
    min_value=0,
    max_value=120
)

# Tenure
check_range(
    "Tenure",
    min_value=0
)

# Balance
check_range(
    "Balance",
    min_value=0
)

# Estimated Salary
check_range(
    "EstimatedSalary",
    min_value=0
)

# Number of Products
check_range(
    "NumOfProducts",
    min_value=1
)


# ============================================================
# 15. NUMERIC-AS-STRING ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("14. NUMERIC-AS-STRING ANALYSIS")
print("=" * 70)

numeric_expected = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

for column in numeric_expected:

    if column not in data.columns:
        continue

    converted = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    invalid_format = (
        data[column].notna()
        & converted.isna()
    )

    count = invalid_format.sum()

    print(
        f"\n--- {column} ---"
    )

    print(
        "Original dtype:",
        data[column].dtype
    )

    print(
        "Non-numeric values:",
        count
    )

    if count > 0:

        print(
            "Examples:"
        )

        print(
            data.loc[
                invalid_format,
                column
            ]
            .head(10)
            .tolist()
        )


# ============================================================
# 16. OUTLIER ANALYSIS - IQR
# ============================================================

print("\n" + "=" * 70)
print("15. OUTLIER ANALYSIS - IQR")
print("=" * 70)

outlier_columns = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary"
]

outlier_results = []

for column in outlier_columns:

    if column not in data.columns:
        continue

    series = pd.to_numeric(
        data[column],
        errors="coerce"
    ).dropna()

    if series.empty:
        continue

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = (
        Q1 - 1.5 * IQR
    )

    upper_bound = (
        Q3 + 1.5 * IQR
    )

    outlier_mask = (
        (series < lower_bound)
        |
        (series > upper_bound)
    )

    outlier_count = (
        outlier_mask.sum()
    )

    outlier_percentage = (
        outlier_count
        / len(series)
        * 100
    )

    outlier_results.append({

        "Feature": column,

        "Q1": Q1,

        "Q3": Q3,

        "IQR": IQR,

        "Lower_Bound":
            lower_bound,

        "Upper_Bound":
            upper_bound,

        "Outlier_Count":
            outlier_count,

        "Outlier_%":
            outlier_percentage
    })


if outlier_results:

    outlier_df = pd.DataFrame(
        outlier_results
    )

    print(
        outlier_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}"
        )
    )


# ============================================================
# 17. BALANCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("16. BALANCE ANALYSIS")
print("=" * 70)

if "Balance" in data.columns:

    balance = pd.to_numeric(
        data["Balance"],
        errors="coerce"
    )

    zero_balance = (
        balance == 0
    ).sum()

    positive_balance = (
        balance > 0
    ).sum()

    print(
        f"Balance = 0: "
        f"{zero_balance:,} "
        f"({zero_balance / len(data) * 100:.2f}%)"
    )

    print(
        f"Balance > 0: "
        f"{positive_balance:,} "
        f"({positive_balance / len(data) * 100:.2f}%)"
    )

    if TARGET in data.columns:

        balance_status = np.where(
            balance == 0,
            "Zero Balance",
            "Positive Balance"
        )

        balance_analysis = pd.DataFrame({

            "BalanceStatus":
                balance_status,

            TARGET:
                data[TARGET]

        })

        print(
            "\nChurn rate by balance:"
        )

        result = (
            balance_analysis
            .groupby(
                "BalanceStatus"
            )[TARGET]
            .agg(
                Customers="count",
                Churn_Rate="mean"
            )
        )

        result["Churn_Rate"] *= 100

        print(
            result.round(2)
        )


# ============================================================
# 18. CHURN RATE BY CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("17. CHURN RATE BY FEATURES")
print("=" * 70)

if TARGET in data.columns:

    analysis_columns = [
        "Geography",
        "Gender",
        "HasCrCard",
        "IsActiveMember",
        "NumOfProducts"
    ]

    for column in analysis_columns:

        if column not in data.columns:
            continue

        print(
            f"\n--- {column} ---"
        )

        result = (
            data
            .groupby(
                column,
                dropna=False
            )[TARGET]
            .agg(
                Customers="count",
                Churn_Rate="mean"
            )
        )

        result["Churn_Rate"] *= 100

        print(
            result.round(2)
        )


# ============================================================
# 19. AGE GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("18. CHURN RATE BY AGE GROUP")
print("=" * 70)

if (
    "Age" in data.columns
    and TARGET in data.columns
):

    age = pd.to_numeric(
        data["Age"],
        errors="coerce"
    )

    age_group = pd.cut(
        age,
        bins=[
            0,
            25,
            35,
            45,
            55,
            65,
            120
        ],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "65+"
        ]
    )

    age_analysis = (
        data
        .assign(
            AgeGroup=age_group
        )
        .groupby(
            "AgeGroup",
            observed=False,
            dropna=False
        )[TARGET]
        .agg(
            Customers="count",
            Churn_Rate="mean"
        )
    )

    age_analysis[
        "Churn_Rate"
    ] *= 100

    print(
        age_analysis.round(2)
    )


# ============================================================
# 20. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("19. CORRELATION WITH TARGET")
print("=" * 70)

if TARGET in data.columns:

    correlation_data = data.copy()

    for column in numeric_expected:

        if column in correlation_data.columns:

            correlation_data[column] = (
                pd.to_numeric(
                    correlation_data[column],
                    errors="coerce"
                )
            )

    numeric_data = (
        correlation_data
        .select_dtypes(
            include=np.number
        )
    )

    if TARGET in numeric_data.columns:

        correlation = (
            numeric_data
            .corr()[TARGET]
            .sort_values(
                ascending=False
            )
        )

        print(
            correlation.round(4)
        )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("20. FINAL DATA AUDIT SUMMARY")
print("=" * 70)

print(
    f"\nDataset shape: "
    f"{data.shape}"
)

print(
    f"Total missing values: "
    f"{data.isna().sum().sum():,}"
)

print(
    f"Duplicate rows: "
    f"{data.duplicated().sum():,}"
)

print(
    "\nColumns containing missing values:"
)

for column in data.columns:

    count = data[column].isna().sum()

    if count > 0:

        percentage = (
            count / len(data) * 100
        )

        print(
            f"- {column}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

print(
    "\nIdentifier columns:"
)

for column in identifier_columns:

    if column in data.columns:

        print(
            f"- {column}"
        )

print(
    f"\nTarget: {TARGET}"
)

print(
    "\nAudit completed."
)

print("=" * 70)