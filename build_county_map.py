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
        height=500,
        dragmode=False,

        margin=dict(
            l=0,
            r=0,
            t=10,
            b=20,
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        coloraxis_showscale=False,

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
    county_stats,
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

    region_data = {
        "Statewide": {
            "display": "Statewide",
            "total": int(statewide["total"]),
            "dem": int(statewide["dem"]),
            "dem_pct": float(statewide["dem_pct"]),
            "rep": int(statewide["rep"]),
            "rep_pct": float(statewide["rep_pct"]),
            "third": int(statewide["third"]),
            "third_pct": float(statewide["third_pct"]),
            "margin_party": statewide["margin_party"],
            "margin_count": int(statewide["margin_count"]),
            "margin_pct": float(statewide["margin_pct"]),
            "signed_margin_pct": float(
                statewide["dem_pct"] - statewide["rep_pct"]
            ),
        }
    }

    for _, row in county_stats.sort_values("county").iterrows():
        county_name = str(row["county"])
        margin = int(row["dem_rep_margin"])

        if margin > 0:
            margin_party = "Democratic"
        elif margin < 0:
            margin_party = "Republican"
        else:
            margin_party = "Even"

        region_data[county_name] = {
            "display": f"{county_name} County",
            "total": int(row["total"]),
            "dem": int(row["dem"]),
            "dem_pct": float(row["dem_pct"]),
            "rep": int(row["rep"]),
            "rep_pct": float(row["rep_pct"]),
            "third": int(row["third_party_unaffiliated"]),
            "third_pct": float(row["third_party_unaffiliated_pct"]),
            "margin_party": margin_party,
            "margin_count": abs(margin),
            "margin_pct": abs(float(row["dem_rep_margin_pct"])),
            "signed_margin_pct": float(
                row["dem_rep_margin_pct"]
            ),
        }

    neighbors_by_county = {}

    for idx, row in county_stats.iterrows():
        county_name = str(row["county"])
        neighbors = []

        for other_idx, other in county_stats.iterrows():
            if idx == other_idx:
                continue

            shared_boundary = (
                row.geometry.boundary
                .intersection(other.geometry.boundary)
            )

            if (
                not shared_boundary.is_empty
                and shared_boundary.length > 1e-9
            ):
                neighbors.append(
                    str(other["county"])
                )

        neighbors_by_county[county_name] = sorted(
            neighbors
        )

    for county_name, neighbors in (
        neighbors_by_county.items()
    ):
        if county_name in region_data:
            region_data[county_name]["neighbors"] = (
                neighbors
            )
    region_data_json = json.dumps(
        region_data,
        ensure_ascii=False,
    )

    selector_options = "\n".join(
        f'<option value="{name}">{data["display"]}</option>'
        for name, data in region_data.items()
    )

    selector_css = """
    .region-selector {
        margin: 0 0 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .region-selector label {
        font-weight: 700;
    }

    .region-selector select {
        min-height: 44px;
        min-width: 220px;
        padding: 8px 36px 8px 12px;
        border: 1px solid #c8c8c8;
        border-radius: 8px;
        background: #fff;
        font: inherit;
        font-size: 16px;
    }
    """

    selector_script = """
<script>
(() => {
    const regionData = __REGION_DATA__;
    const selector = document.getElementById("region-select");
    const cards = document.querySelectorAll(".summary-card");

    if (!selector || cards.length < 5) return;

    const numberFormat = new Intl.NumberFormat("en-US");

    function updateCards() {
        const data = regionData[selector.value];
        if (!data) return;

        const [registered, dem, rep, third, edge] = cards;

        registered.querySelector(".summary-value").textContent =
            numberFormat.format(data.total);
        registered.querySelector(".summary-detail").textContent =
            data.display;

        dem.querySelector(".summary-value").textContent =
            numberFormat.format(data.dem);
        dem.querySelector(".summary-detail").textContent =
            `${data.dem_pct.toFixed(2)}% of voters`;

        rep.querySelector(".summary-value").textContent =
            numberFormat.format(data.rep);
        rep.querySelector(".summary-detail").textContent =
            `${data.rep_pct.toFixed(2)}% of voters`;

        third.querySelector(".summary-value").textContent =
            numberFormat.format(data.third);
        third.querySelector(".summary-detail").textContent =
            `${data.third_pct.toFixed(2)}% of voters`;

        edge.querySelector(".summary-label").textContent =
            data.display === "Statewide"
                ? "Statewide registration edge"
                : `${data.display} registration edge`;

        edge.querySelector(".summary-value").textContent =
            data.margin_party;

        edge.querySelector(".summary-detail").textContent =
            `${numberFormat.format(data.margin_count)} voters \u00b7 ${data.margin_pct.toFixed(2)} points`;
    }

    selector.addEventListener("change", updateCards);
})();
</script>
""".replace("__REGION_DATA__", region_data_json)

    analysis_css = """
    .analysis-panel {
        max-width: 1040px;
        margin: 18px auto 0;
        padding: 20px;
        background: #ffffff;
        border: 1px solid #d9e0e7;
        border-radius: 12px;
    }

    .analysis-heading-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }

    .analysis-heading-row h2 {
        margin: 0;
        font-size: 1.3rem;
    }

    .analysis-summary {
        margin: 0 0 18px;
        line-height: 1.6;
    }

    .analysis-subhead {
        margin: 20px 0 10px;
        font-size: 1.05rem;
    }

    .rank-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
    }

    .rank-item {
        padding: 12px;
        border: 1px solid #e0e5ea;
        border-radius: 8px;
        background: #f8fafc;
    }

    .rank-label {
        display: block;
        margin-bottom: 4px;
        color: #5d6976;
        font-size: 0.8rem;
    }

    .rank-value {
        font-weight: 700;
        line-height: 1.3;
    }

    .analysis-highlights {
        margin: 8px 0 0;
        padding-left: 22px;
        line-height: 1.6;
    }

    .neighbor-intro {
        line-height: 1.6;
        margin: 0 0 12px;
    }

    .neighbor-note {
        color: #5d6976;
        font-size: 0.82rem;
        margin: 8px 0 0;
    }

    .neighbor-table-wrap {
        width: 100%;
        overflow-x: auto;
    }

    .neighbor-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        white-space: nowrap;
    }

    .neighbor-table th,
    .neighbor-table td {
        padding: 9px 10px;
        border-bottom: 1px solid #e0e5ea;
        text-align: right;
    }

    .neighbor-table th:first-child,
    .neighbor-table td:first-child {
        text-align: left;
    }

    .neighbor-table th {
        color: #5d6976;
        font-size: 0.78rem;
    }

    .neighbor-table .selected-row {
        font-weight: 700;
        background: #f3f6f9;
    }

    .copy-analysis {
        min-height: 44px;
        padding: 9px 15px;
        border: 1px solid #aeb8c2;
        border-radius: 8px;
        background: #ffffff;
        font: inherit;
        font-weight: 600;
        cursor: pointer;
    }

    .copy-analysis:hover {
        background: #f4f6f8;
    }

    .copy-status {
        min-height: 1em;
        margin: 10px 0 0;
        color: #5d6976;
        font-size: 0.82rem;
    }

    @media (max-width: 700px) {
        .analysis-panel {
            padding: 15px;
        }

        .rank-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 430px) {
        .rank-grid {
            grid-template-columns: 1fr;
        }
    }
    """

    analysis_script = """
<script>
(() => {
    const regionData = __REGION_DATA__;
    const selector =
        document.getElementById("region-select");

    const title =
        document.getElementById("analysis-title");
    const summary =
        document.getElementById("analysis-summary");
    const comparison =
        document.getElementById("analysis-comparison");
    const neighborsBox =
        document.getElementById("analysis-neighbors");
    const copyButton =
        document.getElementById("copy-analysis");
    const copyStatus =
        document.getElementById("copy-status");

    if (
        !selector ||
        !title ||
        !summary ||
        !comparison ||
        !neighborsBox ||
        !copyButton
    ) {
        return;
    }

    const numberFormat =
        new Intl.NumberFormat("en-US");

    const countyEntries =
        Object.entries(regionData).filter(
            ([name]) => name !== "Statewide"
        );

    let copyText = "";

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function ordinal(number) {
        const n = Number(number);
        const mod100 = n % 100;

        if (mod100 >= 11 && mod100 <= 13) {
            return `${n}th`;
        }

        switch (n % 10) {
            case 1:
                return `${n}st`;
            case 2:
                return `${n}nd`;
            case 3:
                return `${n}rd`;
            default:
                return `${n}th`;
        }
    }

    function rankDescending(metric, value) {
        return 1 + countyEntries.filter(
            ([, data]) =>
                Number(data[metric]) >
                Number(value)
        ).length;
    }

    function partySubject(party) {
        if (party === "Democratic") {
            return "Democrats";
        }

        if (party === "Republican") {
            return "Republicans";
        }

        return null;
    }

    function edgeSentence(data, statewide=false) {
        if (Number(data.margin_count) === 0) {
            return statewide
                ? "Democratic and Republican registration is even statewide."
                : "Democratic and Republican registration is even.";
        }

        const subject =
            partySubject(data.margin_party);

        const location =
            statewide ? " statewide" : "";

        return (
            `${subject} hold a registration edge${location} ` +
            `of ${numberFormat.format(data.margin_count)} voters, ` +
            `or ${Number(data.margin_pct).toFixed(2)} ` +
            `percentage points.`
        );
    }

    function shortEdge(data) {
        if (Number(data.margin_count) === 0) {
            return "Even";
        }

        const letter =
            data.margin_party === "Democratic"
                ? "D"
                : "R";

        return (
            `${letter} +` +
            `${Number(data.margin_pct).toFixed(2)}`
        );
    }

    function maxBy(metric) {
        return countyEntries.reduce(
            (best, current) =>
                Number(current[1][metric]) >
                Number(best[1][metric])
                    ? current
                    : best
        );
    }

    function closestMargin() {
        return countyEntries.reduce(
            (best, current) =>
                Math.abs(
                    Number(
                        current[1].signed_margin_pct
                    )
                ) <
                Math.abs(
                    Number(
                        best[1].signed_margin_pct
                    )
                )
                    ? current
                    : best
        );
    }

    function renderStatewide(data) {
        title.textContent = "Statewide analysis";

        const text =
            `Pennsylvania has ` +
            `${numberFormat.format(data.total)} ` +
            `registered voters. Democrats account for ` +
            `${numberFormat.format(data.dem)} voters ` +
            `(${Number(data.dem_pct).toFixed(2)}%), ` +
            `Republicans account for ` +
            `${numberFormat.format(data.rep)} ` +
            `(${Number(data.rep_pct).toFixed(2)}%), ` +
            `and ${numberFormat.format(data.third)} ` +
            `voters (${Number(data.third_pct).toFixed(2)}%) ` +
            `are registered with a third party or are ` +
            `unaffiliated. ${edgeSentence(data, true)}`;

        summary.textContent = text;

        const largest = maxBy("total");
        const demHigh = maxBy("dem_pct");
        const repHigh = maxBy("rep_pct");
        const thirdHigh = maxBy("third_pct");
        const closest = closestMargin();

        comparison.innerHTML = `
            <h3 class="analysis-subhead">
                County highlights
            </h3>

            <ul class="analysis-highlights">
                <li>
                    <strong>Largest registered electorate:</strong>
                    ${escapeHtml(largest[1].display)}
                    (${numberFormat.format(largest[1].total)} voters)
                </li>

                <li>
                    <strong>Highest Democratic registration share:</strong>
                    ${escapeHtml(demHigh[1].display)}
                    (${Number(demHigh[1].dem_pct).toFixed(2)}%)
                </li>

                <li>
                    <strong>Highest Republican registration share:</strong>
                    ${escapeHtml(repHigh[1].display)}
                    (${Number(repHigh[1].rep_pct).toFixed(2)}%)
                </li>

                <li>
                    <strong>Highest • Third-party/unaffiliated share:</strong>
                    ${escapeHtml(thirdHigh[1].display)}
                    (${Number(thirdHigh[1].third_pct).toFixed(2)}%)
                </li>

                <li>
                    <strong>Closest Democratic-Republican split:</strong>
                    ${escapeHtml(closest[1].display)}
                    (${shortEdge(closest[1])} points)
                </li>
            </ul>
        `;

        neighborsBox.innerHTML = "";

        copyText =
            `${text}\n\nCounty highlights:\n` +
            `Largest registered electorate: ` +
            `${largest[1].display} ` +
            `(${numberFormat.format(largest[1].total)} voters)\n` +
            `Highest Democratic registration share: ` +
            `${demHigh[1].display} ` +
            `(${Number(demHigh[1].dem_pct).toFixed(2)}%)\n` +
            `Highest Republican registration share: ` +
            `${repHigh[1].display} ` +
            `(${Number(repHigh[1].rep_pct).toFixed(2)}%)\n` +
            `Highest • Third-party/unaffiliated share: ` +
            `${thirdHigh[1].display} ` +
            `(${Number(thirdHigh[1].third_pct).toFixed(2)}%)\n` +
            `Closest Democratic-Republican split: ` +
            `${closest[1].display} ` +
            `(${shortEdge(closest[1])} points)`;
    }

    function renderCounty(name, data) {
        title.textContent =
            `${data.display} analysis`;

        const text =
            `${data.display} has ` +
            `${numberFormat.format(data.total)} ` +
            `registered voters. Democrats account for ` +
            `${numberFormat.format(data.dem)} voters ` +
            `(${Number(data.dem_pct).toFixed(2)}%), ` +
            `Republicans account for ` +
            `${numberFormat.format(data.rep)} ` +
            `(${Number(data.rep_pct).toFixed(2)}%), ` +
            `and ${numberFormat.format(data.third)} ` +
            `voters (${Number(data.third_pct).toFixed(2)}%) ` +
            `are registered with a third party or are ` +
            `unaffiliated. ${edgeSentence(data)}`;

        summary.textContent = text;

        const totalRank =
            rankDescending("total", data.total);

        const demRank =
            rankDescending(
                "dem_pct",
                data.dem_pct
            );

        const repRank =
            rankDescending(
                "rep_pct",
                data.rep_pct
            );

        const thirdRank =
            rankDescending(
                "third_pct",
                data.third_pct
            );

        const balanceRank =
            rankDescending(
                "signed_margin_pct",
                data.signed_margin_pct
            );

        comparison.innerHTML = `
            <h3 class="analysis-subhead">
                How this county compares statewide
            </h3>

            <ul class="analysis-highlights">
                <li>
                    <strong>Registered voters:</strong>
                    ${ordinal(totalRank)} of 67
                </li>
                <li>
                    <strong>Democratic share:</strong>
                    ${ordinal(demRank)} of 67
                </li>
                <li>
                    <strong>Republican share:</strong>
                    ${ordinal(repRank)} of 67
                </li>
                <li>
                    <strong>Third-party/unaffiliated share:</strong>
                    ${ordinal(thirdRank)} of 67
                </li>
                <li>
                    <strong>D-R margin order:</strong>
                    ${ordinal(balanceRank)} of 67
                    from most Democratic
                </li>
            </ul>
        `;

        const neighborNames =
            Array.isArray(data.neighbors)
                ? data.neighbors
                : [];

        const neighbors =
            neighborNames
                .map(neighborName => [
                    neighborName,
                    regionData[neighborName]
                ])
                .filter(([, neighborData]) =>
                    Boolean(neighborData)
                );

        if (!neighbors.length) {
            neighborsBox.innerHTML = `
                <h3 class="analysis-subhead">
                    Pennsylvania neighbors
                </h3>
                <p class="neighbor-intro">
                    No Pennsylvania neighboring counties
                    were identified.
                </p>
            `;

            copyText =
                `${text}\n\nStatewide rankings:\n` +
                `• Registered voters: ${ordinal(totalRank)} of 67\n` +
                `• Democratic share: ${ordinal(demRank)} of 67\n` +
                `• Republican share: ${ordinal(repRank)} of 67\n` +
                `• Third-party/unaffiliated share: ` +
                `${ordinal(thirdRank)} of 67\n` +
                `• D-R margin order: ${ordinal(balanceRank)} of 67 from most Democratic`;

            return;
        }

        const selectedMargin =
            Number(data.signed_margin_pct);

        const moreDemocraticThan =
            neighbors.filter(
                ([, neighbor]) =>
                    selectedMargin >
                    Number(
                        neighbor.signed_margin_pct
                    )
            ).length;

        const moreRepublicanThan =
            neighbors.filter(
                ([, neighbor]) =>
                    selectedMargin <
                    Number(
                        neighbor.signed_margin_pct
                    )
            ).length;

        const localGroup = [
            [name, data],
            ...neighbors
        ];

        const localThirdRank =
            1 + localGroup.filter(
                ([, county]) =>
                    Number(county.third_pct) >
                    Number(data.third_pct)
            ).length;

        const rows = [
            [name, data],
            ...neighbors
        ].map(([countyName, county]) => {
            const selected =
                countyName === name;

            return `
                <tr class="${selected ? "selected-row" : ""}">
                    <td>
                        ${escapeHtml(county.display)}
                    </td>
                    <td>
                        ${numberFormat.format(county.total)}
                    </td>
                    <td>
                        ${Number(county.dem_pct).toFixed(2)}%
                    </td>
                    <td>
                        ${Number(county.rep_pct).toFixed(2)}%
                    </td>
                    <td>
                        ${Number(county.third_pct).toFixed(2)}%
                    </td>
                    <td>
                        ${shortEdge(county)}
                    </td>
                </tr>
            `;
        }).join("");

        const neighborWord =
            neighbors.length === 1
                ? "county"
                : "counties";

        const groupSize =
            neighbors.length + 1;

        const neighborNarrative =
            `${data.display} shares a Pennsylvania ` +
            `county boundary with ${neighbors.length} ` +
            `${neighborWord}. Its Democratic-Republican ` +
            `registration margin is more Democratic than ` +
            `${moreDemocraticThan} of those counties and ` +
            `more Republican than ${moreRepublicanThan}. ` +
            `Its third-party/unaffiliated share ranks ` +
            `${ordinal(localThirdRank)} among the ` +
            `${groupSize}-county comparison group.`;

        neighborsBox.innerHTML = `
            <h3 class="analysis-subhead">
                Compared with neighboring counties
            </h3>

            <p class="neighbor-intro">
                ${escapeHtml(neighborNarrative)}
            </p>

            <div class="neighbor-table-wrap">
                <table class="neighbor-table">
                    <thead>
                        <tr>
                            <th>County</th>
                            <th>Registered</th>
                            <th>Dem.</th>
                            <th>Rep.</th>
                            <th>Third/unaff.</th>
                            <th>D-R edge</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${rows}
                    </tbody>
                </table>
            </div>

            <p class="neighbor-note">
                Pennsylvania neighbors are determined
                from the same U.S. Census county
                boundaries used for the map.
                Out-of-state counties are not included.
            </p>
        `;

        const neighborLines =
            neighbors.map(
                ([, neighbor]) =>
                    `${neighbor.display}: ` +
                    `${Number(neighbor.dem_pct).toFixed(2)}% Democratic, ` +
                    `${Number(neighbor.rep_pct).toFixed(2)}% Republican, ` +
                    `${Number(neighbor.third_pct).toFixed(2)}% third-party/unaffiliated, ` +
                    `${shortEdge(neighbor)} point D-R edge`
            ).join("\\n");

        copyText =
            `${text}\n\nStatewide rankings:\n` +
            `• Registered voters: ${ordinal(totalRank)} of 67\n` +
            `• Democratic share: ${ordinal(demRank)} of 67\n` +
            `• Republican share: ${ordinal(repRank)} of 67\n` +
            `• Third-party/unaffiliated share: ` +
            `${ordinal(thirdRank)} of 67\n` +
            `• D-R margin order: ${ordinal(balanceRank)} of 67 from most Democratic\n\n` +
            `${neighborNarrative}\n\n` +
            `Neighboring counties:\n${neighborLines}`;
    }

    function renderAnalysis() {
        const name = selector.value;
        const data = regionData[name];

        if (!data) {
            return;
        }

        if (name === "Statewide") {
            renderStatewide(data);
        } else {
            renderCounty(name, data);
        }
    }

    async function copyAnalysis() {
        if (!copyText) {
            return;
        }

        try {
            if (
                navigator.clipboard &&
                window.isSecureContext
            ) {
                await navigator.clipboard.writeText(
                    copyText
                );
            } else {
                const textarea =
                    document.createElement("textarea");

                textarea.value = copyText;
                textarea.style.position = "absolute";
                textarea.style.left = "-9999px";

                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            }

            copyStatus.textContent =
                "Analysis copied.";
        } catch (error) {
            copyStatus.textContent =
                "Could not copy automatically.";
        }

        window.setTimeout(() => {
            copyStatus.textContent = "";
        }, 2500);
    }

    selector.addEventListener(
        "change",
        renderAnalysis
    );

    copyButton.addEventListener(
        "click",
        copyAnalysis
    );

    renderAnalysis();
})();
</script>
""".replace(
        "__REGION_DATA__",
        region_data_json,
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
    border: 0;
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
    border: 0;
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
    border: 0;
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
            500px
        ) !important;
}}

.map-content {{
    position: relative;
    width: 100%;
}}

.map-legend {{
    position: absolute;
    right: -22px;
    top: 15%;
    transform: none;
    z-index: 3;

    display: flex;
    flex-direction: column;
    align-items: center;

    padding: 7px 6px;
    background: transparent;
    border: 0;
    border-radius: 8px;

    pointer-events: none;
}}

.map-legend-title {{
    margin-bottom: 7px;
    font-size: 0.66rem;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
}}

.map-legend-scale {{
    display: flex;
    gap: 6px;
    height: 250px;
}}

.map-legend-bar {{
    width: 10px;
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(
        to bottom,
        {BLUE} 0%,
        {PURPLE} 50%,
        {RED} 100%
    );
}}

.map-legend-labels {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-size: 0.66rem;
    line-height: 1;
    white-space: nowrap;
}}

@media (max-width: 700px) {{

    .map-legend {{
        position: static;
        transform: none;

        width: min(92%, 520px);
        margin: 0 auto 6px;
        padding: 0;

        background: transparent;
        border: 0;
    }}

    .map-legend-title {{
        margin-bottom: 5px;
    }}

    .map-legend-scale {{
        width: 100%;
        height: auto;
        flex-direction: column;
        gap: 5px;
    }}

    .map-legend-bar {{
        width: 100%;
        height: 12px;
        background: linear-gradient(
            to right,
            {RED} 0%,
            {PURPLE} 50%,
            {BLUE} 100%
        );
    }}

    .map-legend-labels {{
        width: 100%;
        flex-direction: row-reverse;
        justify-content: space-between;
        font-size: 0.66rem;
    }}
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

{selector_css}
{analysis_css}
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

        <div class="map-content">

            <div class="map-wrap">
                {map_html}
            </div>

            <div
                class="map-legend"
                aria-label="Registration balance legend"
            >
                <div class="map-legend-title">
                    Registration balance
                </div>

                <div class="map-legend-scale">
                    <div class="map-legend-bar"></div>

                    <div class="map-legend-labels">
                        <span>Strong D</span>
                        <span>D lean</span>
                        <span>Even</span>
                        <span>R lean</span>
                        <span>Strong R</span>
                    </div>
                </div>
            </div>

        </div>

    </section>


    <section class="region-selector" aria-label="Registration area">
        <label for="region-select">View registration for:</label>
        <select id="region-select">
            {selector_options}
        </select>
    </section>

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


    <section
        class="analysis-panel"
        aria-labelledby="analysis-title"
    >

        <div class="analysis-heading-row">

            <h2 id="analysis-title">
                Statewide analysis
            </h2>

            <button
                id="copy-analysis"
                class="copy-analysis"
                type="button"
            >
                Copy analysis
            </button>

        </div>

        <p
            id="analysis-summary"
            class="analysis-summary"
        ></p>

        <div id="analysis-comparison"></div>

        <div id="analysis-neighbors"></div>

        <p
            id="copy-status"
            class="copy-status"
            aria-live="polite"
        ></p>

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

{selector_script}
{analysis_script}

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
        merged,
    )

    print(
        "\nDone."
    )











