from pathlib import Path
from datetime import datetime
import json
import re

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_FILE = Path("data/raw/currentvotestats.xlsx")

PROCESSED_DIR = Path("data/processed")

HISTORY_DIR = Path("data/history")

OUTPUT_FILE = (
    PROCESSED_DIR
    / "county_stats_current.csv"
)

METADATA_FILE = (
    PROCESSED_DIR
    / "county_stats_metadata.json"
)


# ---------------------------------------------------------
# Read source-date metadata
# ---------------------------------------------------------

def read_source_date():
    raw_header = pd.read_excel(
        RAW_FILE,
        sheet_name="Reg Voter",
        header=None,
        nrows=1,
    )

    first_cell = str(
        raw_header.iloc[0, 0]
    ).strip()

    match = re.search(
        r"(\d{1,2}/\d{1,2}/\d{4})",
        first_cell,
    )

    if not match:
        raise ValueError(
            "Could not find source date in "
            f"workbook header: {first_cell}"
        )

    source_date = datetime.strptime(
        match.group(1),
        "%m/%d/%Y",
    )

    return source_date


# ---------------------------------------------------------
# Load and clean county registration data
# ---------------------------------------------------------

def load_county_stats():
    df = pd.read_excel(
        RAW_FILE,
        sheet_name="Reg Voter",
        header=1,
    )

    # Remove completely empty rows and columns
    df = df.dropna(
        axis=0,
        how="all",
    )

    df = df.dropna(
        axis=1,
        how="all",
    )

    # Keep only actual county rows
    df = df[
        df["CountyName"].notna()
        & df["CountyID"].notna()
    ].copy()

    # Rename source columns
    df = df.rename(
        columns={
            "CountyName": "county",
            "CountyID": "county_id",
            "Dem": "dem",
            "Rep": "rep",
            "No Aff": "no_aff",
            "Other": "other",
            "Total Count of All Voters": "total",
        }
    )

    # Clean county names
    df["county"] = (
        df["county"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Convert numeric fields
    numeric_columns = [
        "county_id",
        "dem",
        "rep",
        "no_aff",
        "other",
        "total",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).astype("Int64")

    # -----------------------------------------------------
    # Combine third-party and unaffiliated voters
    # -----------------------------------------------------

    df["third_party_unaffiliated"] = (
        df["no_aff"]
        + df["other"]
    )

    df = df.drop(
        columns=[
            "no_aff",
            "other",
        ]
    )

    # -----------------------------------------------------
    # Registration percentages
    # -----------------------------------------------------

    df["dem_pct"] = (
        df["dem"]
        / df["total"]
        * 100
    )

    df["rep_pct"] = (
        df["rep"]
        / df["total"]
        * 100
    )

    df["third_party_unaffiliated_pct"] = (
        df["third_party_unaffiliated"]
        / df["total"]
        * 100
    )

    # -----------------------------------------------------
    # Democratic vs. Republican margin
    #
    # Positive = Democratic advantage
    # Negative = Republican advantage
    # -----------------------------------------------------

    df["dem_rep_margin"] = (
        df["dem"]
        - df["rep"]
    )

    df["dem_rep_margin_pct"] = (
        df["dem_pct"]
        - df["rep_pct"]
    )

    percentage_columns = [
        "dem_pct",
        "rep_pct",
        "third_party_unaffiliated_pct",
        "dem_rep_margin_pct",
    ]

    df[percentage_columns] = (
        df[percentage_columns]
        .round(2)
    )

    df = (
        df
        .sort_values("county")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------
# Validate data
# ---------------------------------------------------------

def validate_county_stats(df):
    print(
        f"County rows found: {len(df)}"
    )

    print(
        "Duplicate counties: "
        f"{df['county'].duplicated().sum()}"
    )

    print(
        "Missing county IDs: "
        f"{df['county_id'].isna().sum()}"
    )

    calculated_total = (
        df["dem"]
        + df["rep"]
        + df["third_party_unaffiliated"]
    )

    mismatches = df[
        calculated_total != df["total"]
    ]

    print(
        "Rows where registration groups "
        "do not equal total: "
        f"{len(mismatches)}"
    )


# ---------------------------------------------------------
# Save dated historical snapshot
# ---------------------------------------------------------

def save_history_snapshot(
    df,
    source_date,
):
    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_date_key = source_date.strftime(
        "%Y-%m-%d"
    )

    history_file = (
        HISTORY_DIR
        / f"county_stats_{source_date_key}.csv"
    )

    history_metadata_file = (
        HISTORY_DIR
        / f"county_stats_{source_date_key}_metadata.json"
    )

    if history_file.exists():
        print(
            f"Historical snapshot already exists: "
            f"{history_file}"
        )
        return

    df.to_csv(
        history_file,
        index=False,
    )

    month = source_date.strftime("%b.")
    day = source_date.day
    year = source_date.year

    metadata = {
        "source_date": source_date_key,
        "source_date_display": f"{month} {day}, {year}",
        "county_count": len(df),
        "source": "Pennsylvania Department of State",
    }

    history_metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"Saved historical snapshot: "
        f"{history_file}"
    )

    print(
        f"Saved historical metadata: "
        f"{history_metadata_file}"
    )


# ---------------------------------------------------------
# Save cleaned county data
# ---------------------------------------------------------

def save_county_stats(df):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved cleaned county data: "
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

def save_metadata(
    source_date,
    county_count,
):
    month = source_date.strftime("%b.")
    day = source_date.day
    year = source_date.year

    source_date_display = f"{month} {day}, {year}"

    metadata = {
        "source_date": source_date.strftime("%Y-%m-%d"),
        "source_date_display": source_date_display,
        "county_count": county_count,
        "source": "Pennsylvania Department of State",
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved metadata: {METADATA_FILE}")
    print(f"Source data date: {metadata['source_date_display']}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    source_date = read_source_date()

    county_stats = load_county_stats()

    validate_county_stats(
        county_stats
    )

    save_history_snapshot(
        county_stats,
        source_date,
    )

    save_county_stats(
        county_stats
    )

    save_metadata(
        source_date,
        len(county_stats),
    )

    print(
        "\nCounty comparison preview:"
    )

    print(
        county_stats
        .head(10)
        .to_string(index=False)
    )