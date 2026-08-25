import pandas as pd
# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = r"D:\AI_Projects\ML-Lab\data\raw\Bank_churn.csv"
data = pd.read_csv(DATA_PATH)
data.drop(columns = "Surname", inplace= True)
# ============================================================
# BASIC INFORMATION
# ============================================================

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nShape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nDtypes:")
print(data.dtypes)

print("\nHead:")
print(data.head())

print("\nInfo:")
data.info()

# ============================================================
# MISSING VALUE
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE")
print("=" * 60)

missing = data.isna().sum()
missing_rate = (
    data.isna()
    .mean()
    .sort_values(ascending=False)
)

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_rate": missing_rate
})

print(missing_summary)

# ============================================================
# DUPLICATE
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE")
print("=" * 60)

duplicate_count = data.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_count)

print("\nDuplicate rows:")
print(data[data.duplicated(keep=False)])

# ============================================================
# REMOVE IRRELEVANT COLUMN
# ============================================================

if "Surname" in data.columns:
    data.drop(columns=["Surname"], inplace=True)

print("\nShape after dropping Surname:")
print(data.shape)


# ============================================================
# STATISTICAL DESCRIPTION
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL DESCRIPTION")
print("=" * 60)
print(data.describe().T)


print("\n" + "=" * 60)
print("CATEGORICAL DESCRIPTION")
print("=" * 60)

categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns

if len(categorical_columns) > 0:
    print(data[categorical_columns].describe().T)