from pathlib import Path
import pandas as pd

SOURCE_FILE = Path(
    "data/reference/Official_8312026125214PM.CSV"
)

OUTPUT_FILE = Path(
    "data/reference/presidential_2024_county_results.csv"
)

df = pd.read_csv(SOURCE_FILE)

df["Votes"] = (
    df["Votes"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(int)
)

candidate_map = {
    "KAMALA D HARRIS": "harris_votes",
    "DONALD J TRUMP": "trump_votes",
    "CHASE OLIVER": "oliver_votes",
    "JILL STEIN": "stein_votes",
}

df = df[
    df["Candidate Name"].isin(candidate_map)
].copy()

df["vote_column"] = (
    df["Candidate Name"]
    .map(candidate_map)
)

results = (
    df.pivot(
        index="County Name",
        columns="vote_column",
        values="Votes",
    )
    .reset_index()
    .rename(
        columns={
            "County Name": "county",
        }
    )
)

results["county"] = (
    results["county"]
    .str.title()
)

# Correct title-casing for McKean.
results["county"] = results["county"].replace(
    {"Mckean": "McKean"}
)

vote_columns = [
    "harris_votes",
    "trump_votes",
    "oliver_votes",
    "stein_votes",
]

results["total_presidential_votes"] = (
    results[vote_columns].sum(axis=1)
)

results["winner"] = results.apply(
    lambda row:
        "Trump"
        if row["trump_votes"] > row["harris_votes"]
        else "Harris",
    axis=1,
)

results["winner_votes"] = results[
    ["trump_votes", "harris_votes"]
].max(axis=1)

results["margin_votes"] = (
    results["trump_votes"]
    - results["harris_votes"]
).abs()

results["winner_pct"] = (
    results["winner_votes"]
    / results["total_presidential_votes"]
    * 100
).round(2)

results["margin_pct"] = (
    results["margin_votes"]
    / results["total_presidential_votes"]
    * 100
).round(2)

results = results[
    [
        "county",
        "harris_votes",
        "trump_votes",
        "oliver_votes",
        "stein_votes",
        "total_presidential_votes",
        "winner",
        "winner_votes",
        "margin_votes",
        "winner_pct",
        "margin_pct",
    ]
].sort_values("county")

if len(results) != 67:
    raise ValueError(
        f"Expected 67 counties, found {len(results)}"
    )

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"Saved: {OUTPUT_FILE}"
)

print(
    f"Counties: {len(results)}"
)

print(
    "\nBucks County:"
)

print(
    results[
        results["county"] == "Bucks"
    ].to_string(index=False)
)
