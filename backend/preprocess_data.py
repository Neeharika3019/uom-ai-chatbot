import pandas as pd
import json
import os

PROGRAMMES_FILE = "data/working/programmes.xlsx"
ADMISSIONS_FILE = "data/working/Admission_Faculty.xlsx"

OUTPUT_DIR = "data/processed"

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_dataframe(df, dataset_name):
    print(f"\nCleaning {dataset_name}...")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove fully empty rows
    df = df.dropna(how="all")

    # Replace NaN values with empty strings
    df = df.fillna("")

    # Clean text values
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.strip()

    # Check duplicate IDs
    if "ID" in df.columns:
        duplicate_ids = df[df["ID"].duplicated()]["ID"].tolist()

        if duplicate_ids:
            print("WARNING - Duplicate IDs found:", duplicate_ids)
        else:
            print("No duplicate IDs found.")

    return df


# Read Excel files
programmes_df = pd.read_excel(PROGRAMMES_FILE)
admissions_df = pd.read_excel(ADMISSIONS_FILE)

# Clean datasets
programmes_df = clean_dataframe(
    programmes_df,
    "Programmes dataset"
)

admissions_df = clean_dataframe(
    admissions_df,
    "Admissions & Faculty dataset"
)


# Keep only verified records
if "Verified Status" in programmes_df.columns:
    programmes_verified = programmes_df[
        programmes_df["Verified Status"].str.lower() == "verified"
    ].copy()
else:
    programmes_verified = programmes_df.copy()


if "Verified Status" in admissions_df.columns:
    admissions_verified = admissions_df[
        admissions_df["Verified Status"].str.lower() == "verified"
    ].copy()
else:
    admissions_verified = admissions_df.copy()


print("\n=== CLEANING SUMMARY ===")
print("Programme records before cleaning:", len(programmes_df))
print("Verified programme records:", len(programmes_verified))

print("Admission/Faculty records before cleaning:", len(admissions_df))
print("Verified Admission/Faculty records:", len(admissions_verified))


# Convert to dictionaries
programmes_records = programmes_verified.to_dict(orient="records")
admissions_records = admissions_verified.to_dict(orient="records")


# Save JSON files
programmes_output = os.path.join(
    OUTPUT_DIR,
    "programmes.json"
)

admissions_output = os.path.join(
    OUTPUT_DIR,
    "admissions_faculty.json"
)


with open(
    programmes_output,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        programmes_records,
        file,
        indent=4,
        ensure_ascii=False
    )


with open(
    admissions_output,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        admissions_records,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\nJSON files created successfully:")
print(programmes_output)
print(admissions_output)