from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_URL = (
    "https://www.pa.gov/content/dam/copapwp-pagov/en/dos/resources/"
    "voting-and-elections/voting-and-election-statistics/"
    "currentvotestats.xlsx"
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

OUTPUT_FILE = RAW_DIR / "currentvotestats.xlsx"


# ---------------------------------------------------------
# Download workbook
# ---------------------------------------------------------

def download_workbook():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading current PA county voter statistics...")

    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()

    OUTPUT_FILE.write_bytes(response.content)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Downloaded {len(response.content):,} bytes")


# ---------------------------------------------------------
# Inspect workbook
# ---------------------------------------------------------

def inspect_workbook():
    excel_file = pd.ExcelFile(OUTPUT_FILE)

    print("\nWorkbook sheets:")

    for sheet in excel_file.sheet_names:
        print(f"  - {sheet}")

    print("\nCurrent county registration data:")

    df = pd.read_excel(
        OUTPUT_FILE,
        sheet_name="Reg Voter",
        header=1,
    )

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 10 counties:")
    print(df.head(10).to_string(index=False))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    download_workbook()
    inspect_workbook()