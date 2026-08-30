from pathlib import Path
import json
import re
import unicodedata
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

STATS_FILE = Path(
    "data/processed/county_stats_current.csv"
)

METADATA_FILE = Path(
    "data/processed/county_stats_metadata.json"
)

RAW_DIR = Path("data/raw")

SHAPE_ZIP = RAW_DIR / "cb_2025_us_county_500k.zip"

SHAPE_DIR = (
    RAW_DIR
    / "cb_2025_us_county_500k"
)

OUTPUT_DIR = Path("docs")

OUTPUT_FILE = (
    OUTPUT_DIR
    / "index.html"
)


# ---------------------------------------------------------
# Sources
# ---------------------------------------------------------

COUNTY_SHAPE_URL = (
    "https://www2.census.gov/geo/tiger/"
    "GENZ2025/shp/"
    "cb_2025_us_county_500k.zip"
)

PA_VOTER_STATS_URL = (
    "https://www.pa.gov/agencies/dos/resources/"
    "voting-and-elections-resources/"
    "voting-and-election-statistics"
)


# ---------------------------------------------------------
# Map configuration
# ---------------------------------------------------------

RED = "#b2182b"
PURPLE = "#7b3294"
BLUE = "#2166ac"

COLOR_SCALE = [
    [0.0, RED],
    [0.5, PURPLE],
    [1.0, BLUE],
]

ASINH_SCALE = 5.0


# ---------------------------------------------------------
# County-name normalization
# ---------------------------------------------------------

def normalize_county_name(value):
    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = text.casefold()

    text = re.sub(
        r"\bcounty\b",
        "",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )

    return text


# ---------------------------------------------------------
# Download Census boundaries
# ---------------------------------------------------------

def download_shapes():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SHAPE_ZIP.exists():
        print(
            f"Boundary file already downloaded: "
            f"{SHAPE_ZIP}"
        )
        return

    print(
        "Downloading Census county boundaries..."
    )

    response = requests.get(
        COUNTY_SHAPE_URL,
        timeout=120,
    )

    response.raise_for_status()

    SHAPE_ZIP.write_bytes(
        response.content
    )

    print(
        f"Saved: {SHAPE_ZIP}"
    )


# ---------------------------------------------------------
# Extract Census boundaries
# ---------------------------------------------------------

def extract_shapes():
    SHAPE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(
        SHAPE_DIR.glob("*.shp")
    )

    if existing:
        print(
            "Boundary shapefile already extracted."
        )
        return

    print(
        "Extracting Census county boundaries..."
    )

    with zipfile.ZipFile(
        SHAPE_ZIP
    ) as archive:

        archive.extractall(
            SHAPE_DIR
        )

    print(
        f"Extracted to: {SHAPE_DIR}"
    )


# ---------------------------------------------------------
# Load voter-registration data
# ---------------------------------------------------------

def load_stats():
    print(
        "\nLoading PA voter-registration data..."
    )

    stats = pd.read_csv(
        STATS_FILE
    )

    stats["county_key"] = (
        stats["county"]
        .apply(normalize_county_name)
    )

    print(
        f"Registration counties: {len(stats)}"
    )

    return stats


# ---------------------------------------------------------
# Load source metadata
# ---------------------------------------------------------

def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Source data date: "
        f"{metadata['source_date_display']}"
    )

    return metadata


# ---------------------------------------------------------
# Load Pennsylvania county geography
# ---------------------------------------------------------

def load_counties():
    shape_files = list(
        SHAPE_DIR.glob("*.shp")
    )

    if not shape_files:
        raise FileNotFoundError(
            "No Census shapefile was found."
        )

    counties = gpd.read_file(
        shape_files[0]
    )

    # Pennsylvania FIPS = 42
    pa = counties[
        counties["STATEFP"] == "42"
    ].copy()

    pa = pa.to_crs(
        epsg=4326
    )

    pa["county_key"] = (
        pa["NAME"]
        .apply(normalize_county_name)
    )

    print(
        f"Census PA county shapes: {len(pa)}"
    )

    return pa


# ---------------------------------------------------------
# Validate county matching
# ---------------------------------------------------------

def validate_names(
    stats,
    counties,
):
    print(
        "\nChecking county names..."
    )

    stats_duplicates = stats[
        stats["county_key"].duplicated(
            keep=False
        )
    ]

    census_duplicates = counties[
        counties["county_key"].duplicated(
            keep=False
        )
    ]

    if not stats_duplicates.empty:
        print(
            "\nDuplicate registration "
            "county keys:"
        )

        print(
            stats_duplicates[
                [
                    "county",
                    "county_key",
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate county keys found "
            "in registration data."
        )

    if not census_duplicates.empty:
        print(
            "\nDuplicate Census county keys:"
        )

        print(
            census_duplicates[
                [
                    "NAME",
                    "county_key",
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate county keys found "
            "in Census data."
        )

    stats_keys = set(
        stats["county_key"]
    )

    census_keys = set(
        counties["county_key"]
    )

    missing_from_map = (
        stats_keys
        - census_keys
    )

    missing_from_stats = (
        census_keys
        - stats_keys
    )

    if missing_from_map:
        print(
            "\nRegistration counties "
            "not found in Census map:"
        )

        for key in sorted(
            missing_from_map
        ):
            row = stats[
                stats["county_key"] == key
            ].iloc[0]

            print(
                f"  - {row['county']}"
            )

    if missing_from_stats:
        print(
            "\nCensus counties "
            "not found in registration data:"
        )

        for key in sorted(
            missing_from_stats
        ):
            row = counties[
                counties["county_key"] == key
            ].iloc[0]

            print(
                f"  - {row['NAME']}"
            )

    if (
        missing_from_map
        or missing_from_stats
    ):
        raise ValueError(
            "County-name matching failed. "
            "Dashboard was not built."
        )

    print(
        "All 67 counties matched."
    )


# ---------------------------------------------------------
# Merge registration data with geography
# ---------------------------------------------------------

def merge_data(
    counties,
    stats,
):
    merged = counties.merge(
        stats,
        on="county_key",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Merged county rows: {len(merged)}"
    )

    differences = merged[
        merged["NAME"]
        != merged["county"]
    ][
        [
            "county",
            "NAME",
        ]
    ]

    if not differences.empty:
        print(
            "\nCounty display-name differences:"
        )

        for _, row in differences.iterrows():
            print(
                f"  Registration: "
                f"{row['county']}"
                f"  ->  Census: "
                f"{row['NAME']}"
            )

    merged["display_county"] = (
        merged["NAME"]
    )

    return merged


# ---------------------------------------------------------
# Calculate map color metric
# ---------------------------------------------------------

def calculate_map_metric(merged):
    print(
        "\nCalculating map color metric..."
    )

    # Third-party/unaffiliated share pulls
    # the D-R margin toward purple.
    merged["adjusted_margin"] = (
        merged["dem_rep_margin_pct"]
        * (
            1
            - (
                merged[
                    "third_party_unaffiliated_pct"
                ]
                / 100
            )
        )
    )

    max_abs_margin = (
        merged["adjusted_margin"]
        .abs()
        .max()
    )

    max_transformed = np.arcsinh(
        max_abs_margin
        / ASINH_SCALE
    )

    merged["color_value"] = (
        np.arcsinh(
            merged["adjusted_margin"]
            / ASINH_SCALE
        )
        / max_transformed
    )

    print(
        f"Maximum adjusted margin: "
        f"{max_abs_margin:.2f}"
    )

    print(
        f"Asinh scale: {ASINH_SCALE:g}"
    )

    return merged


# ---------------------------------------------------------
# Reader-friendly county margin labels
# ---------------------------------------------------------

def add_margin_labels(merged):
    def describe_margin(row):
        margin_count = int(
            row["dem_rep_margin"]
        )

        margin_pct = float(
            row["dem_rep_margin_pct"]
        )

        if margin_count > 0:
            return (
                "Democratic registration edge: "
                f"{margin_count:,} voters "
                f"({abs(margin_pct):.2f} points)"
            )

        if margin_count < 0:
            return (
                "Republican registration edge: "
                f"{abs(margin_count):,} voters "
                f"({abs(margin_pct):.2f} points)"
            )

        return (
            "Democratic and Republican "
            "registration is even"
        )

    merged["margin_label"] = merged.apply(
        describe_margin,
        axis=1,
    )

    return merged


# ---------------------------------------------------------
# Statewide summary
# ---------------------------------------------------------

def calculate_statewide_stats(stats):
    total = int(
        stats["total"].sum()
    )

    dem = int(
        stats["dem"].sum()
    )

    rep = int(
        stats["rep"].sum()
    )

    third = int(
        stats[
            "third_party_unaffiliated"
        ].sum()
    )

    dem_pct = (
        dem
        / total
        * 100
    )

    rep_pct = (
        rep
        / total
        * 100
    )

    third_pct = (
        third
        / total
        * 100
    )

    margin = (
        dem
        - rep
    )

    margin_pct = (
        dem_pct
        - rep_pct
    )

    if margin > 0:
        margin_party = "Democratic"
        margin_count = margin

    elif margin < 0:
        margin_party = "Republican"
        margin_count = abs(margin)

    else:
        margin_party = "Even"
        margin_count = 0

    return {
        "total": total,
        "dem": dem,
        "rep": rep,
        "third": third,
        "dem_pct": dem_pct,
        "rep_pct": rep_pct,
        "third_pct": third_pct,
        "margin_party": margin_party,
        "margin_count": margin_count,
        "margin_pct": abs(margin_pct),
    }


# ---------------------------------------------------------
# Build Plotly county map
# ---------------------------------------------------------

def build_map(merged):
    print(
        "\nBuilding Pennsylvania county map..."
    )

    geojson = merged.__geo_interface__

    fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="GEOID",
        featureidkey="properties.GEOID",
        color="color_value",
        range_color=(-1, 1),
        color_continuous_scale=COLOR_SCALE,

        custom_data=[
            "display_county",
            "dem",
            "dem_pct",
            "rep",
            "rep_pct",
            "third_party_unaffiliated",
            "third_party_unaffiliated_pct",
            "total",
            "margin_label",
        ],
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.7,

        hovertemplate=(
            "<b>%{customdata[0]} County</b>"
            "<br><br>"

            "Democratic: "
            "%{customdata[1]:,} "
            "(%{customdata[2]:.2f}%)"
            "<br>"

            "Republican: "
            "%{customdata[3]:,} "
            "(%{customdata[4]:.2f}%)"
            "<br>"

            "Third-party/unaffiliated: "
            "%{customdata[5]:,} "
            "(%{customdata[6]:.2f}%)"
            "<br>"

            "Total registered: "
            "%{customdata[7]:,}"
            "<br><br>"

            "%{customdata[8]}"

            "<extra></extra>"
        ),
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )

    fig.update_layout(
        height=620,
        dragmode=False,

        margin=dict(
            l=0,
            r=0,
            t=10,
            b=85,
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        coloraxis_colorbar=dict(
            title="Registration balance",
            orientation="h",

            x=0.5,
            xanchor="center",

            y=-0.06,
            yanchor="top",

            len=0.70,
            thickness=14,

            tickvals=[
                -1.0,
                -0.55,
                0,
                0.55,
                1.0,
            ],

            ticktext=[
                "Strong R",
                "R lean",
                "Even",
                "D lean",
                "Strong D",
            ],
        ),
    )

    return fig


# ---------------------------------------------------------
# Build complete dashboard HTML
# ---------------------------------------------------------

def build_dashboard(
    fig,
    statewide,
    metadata,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": False,
        },
    )

    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
Pennsylvania Voter Registration Dashboard
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f4f5f7;
    color: #1f2933;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
}}

.page {{
    width: 100%;
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 20px 40px;
}}

.header {{
    margin-bottom: 22px;
}}

.header h1 {{
    margin: 0 0 8px;
    font-size: clamp(
        1.65rem,
        4vw,
        2.35rem
    );
    line-height: 1.15;
}}

.subtitle {{
    margin: 0;
    color: #586574;
    font-size: 1rem;
    line-height: 1.5;
}}

.data-date {{
    margin: 7px 0 0;
    color: #647180;
    font-size: 0.9rem;
    font-weight: 600;
}}

.summary-grid {{
    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(175px, 1fr)
        );

    gap: 12px;
    margin-bottom: 20px;
}}

.summary-card {{
    background: white;
    border: 1px solid #d9dee5;
    border-radius: 10px;
    padding: 15px 16px;
}}

.summary-label {{
    color: #647180;
    font-size: 0.85rem;
    margin-bottom: 7px;
}}

.summary-value {{
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.1;
}}

.summary-detail {{
    margin-top: 6px;
    color: #4e5b68;
    font-size: 0.88rem;
}}

.explainer {{
    background: #ffffff;
    border: 1px solid #d9dee5;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 20px;
}}

.explainer h2 {{
    margin: 0 0 8px;
    font-size: 1.05rem;
}}

.explainer p {{
    margin: 0;
    line-height: 1.55;
    color: #465463;
}}

.map-panel {{
    max-width: 1040px;
    margin: 0 auto;

    background: white;
    border: 1px solid #d9dee5;
    border-radius: 12px;

    padding: 18px 18px 10px;
}}

.map-header {{
    margin-bottom: 4px;
}}

.map-header h2 {{
    margin: 0 0 5px;
    font-size: 1.25rem;
}}

.map-header p {{
    margin: 0;
    color: #647180;
    font-size: 0.9rem;
    line-height: 1.4;
}}

.map-wrap {{
    width: 100%;
    min-width: 0;
}}

.map-wrap .plotly-graph-div {{
    width: 100% !important;

    height:
        clamp(
            360px,
            55vw,
            620px
        ) !important;
}}

.methodology {{
    max-width: 1040px;
    margin: 14px auto 0;

    color: #5d6976;
    font-size: 0.84rem;
    line-height: 1.5;
}}

.methodology p {{
    margin: 5px 0;
}}

.methodology a {{
    color: #315f94;
}}

@media (
    max-width: 700px
) {{

    .page {{
        padding: 18px 12px 28px;
    }}

    .summary-grid {{
        grid-template-columns:
            repeat(
                2,
                minmax(0, 1fr)
            );
    }}

    .summary-card {{
        padding: 13px;
    }}

    .summary-value {{
        font-size: 1.25rem;
    }}

    .map-panel {{
        padding: 14px 8px 8px;
    }}

}}

@media (
    max-width: 430px
) {{

    .summary-grid {{
        grid-template-columns: 1fr;
    }}

    .explainer {{
        padding: 14px;
    }}

    .map-wrap .plotly-graph-div {{
        height: 360px !important;
    }}

}}

</style>

</head>

<body>

<main class="page">

    <header class="header">

        <h1>
            Pennsylvania voter registration
        </h1>

        <p class="subtitle">
            Current voter registration totals
            and party balance across Pennsylvania's
            67 counties.
        </p>

        <p class="data-date">
            Data from PA Dept. of State last updated {metadata["source_date_display"]}
        </p>

    </header>


    <section class="summary-grid">

        <div class="summary-card">

            <div class="summary-label">
                Registered voters
            </div>

            <div class="summary-value">
                {statewide["total"]:,}
            </div>

            <div class="summary-detail">
                Statewide
            </div>

        </div>


        <div class="summary-card">

            <div class="summary-label">
                Democratic
            </div>

            <div class="summary-value">
                {statewide["dem"]:,}
            </div>

            <div class="summary-detail">
                {statewide["dem_pct"]:.2f}% of voters
            </div>

        </div>


        <div class="summary-card">

            <div class="summary-label">
                Republican
            </div>

            <div class="summary-value">
                {statewide["rep"]:,}
            </div>

            <div class="summary-detail">
                {statewide["rep_pct"]:.2f}% of voters
            </div>

        </div>


        <div class="summary-card">

            <div class="summary-label">
                Third-party/unaffiliated
            </div>

            <div class="summary-value">
                {statewide["third"]:,}
            </div>

            <div class="summary-detail">
                {statewide["third_pct"]:.2f}% of voters
            </div>

        </div>


        <div class="summary-card">

            <div class="summary-label">
                Statewide registration edge
            </div>

            <div class="summary-value">
                {statewide["margin_party"]}
            </div>

            <div class="summary-detail">
                {statewide["margin_count"]:,} voters
                &middot; {statewide["margin_pct"]:.2f} points
            </div>

        </div>

    </section>


    <section class="explainer">

        <h2>
            How to read the map
        </h2>

        <p>
            Blue counties have a Democratic
            registration advantage and red counties
            have a Republican advantage. Counties
            closer to purple have a more balanced
            Democratic-Republican registration split
            or a larger share of third-party and
            unaffiliated voters. Hover over a county
            for the actual registration totals and
            Democratic-Republican margin.
        </p>

    </section>


    <section class="map-panel">

        <div class="map-header">

            <h2>
                County registration balance
            </h2>

            <p>
                Color shows registration balance;
                it does not represent election results
                or a forecast.
            </p>

        </div>

        <div class="map-wrap">
            {map_html}
        </div>

    </section>


    <section class="methodology">

        <p>
            <strong>About the colors:</strong>
            Democratic-Republican registration
            determines the red or blue direction.
            The third-party/unaffiliated share pulls
            the color toward purple. A nonlinear
            color scale gives greater visual
            separation to counties with relatively
            close registration margins.
        </p>

        <p>
            Data as of
            {metadata["source_date_display"]}.
            Source:
            <a
                href="{PA_VOTER_STATS_URL}"
                target="_blank"
                rel="noopener noreferrer"
            >
                Pennsylvania Department of State
            </a>.
            County boundaries are from the
            U.S. Census Bureau.
        </p>

    </section>

</main>

</body>

</html>
"""

    OUTPUT_FILE.write_text(
        html,
        encoding="utf-8",
    )

    print(
        f"\nDashboard saved: "
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    download_shapes()

    extract_shapes()

    stats = load_stats()

    metadata = load_metadata()

    counties = load_counties()

    validate_names(
        stats,
        counties,
    )

    merged = merge_data(
        counties,
        stats,
    )

    merged = calculate_map_metric(
        merged
    )

    merged = add_margin_labels(
        merged
    )

    statewide = (
        calculate_statewide_stats(
            stats
        )
    )

    fig = build_map(
        merged
    )

    build_dashboard(
        fig,
        statewide,
        metadata,
    )

    print(
        "\nDone."
    )


