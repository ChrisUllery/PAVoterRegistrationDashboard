from pathlib import Path
import subprocess
import sys
from datetime import datetime


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "download_county_stats.py",
    "process_county_stats.py",
    "build_county_map.py",
]


# ---------------------------------------------------------
# Run one script
# ---------------------------------------------------------

def run_script(script_name):
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script not found: {script_path}"
        )

    print()
    print("=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with "
            f"exit code {result.returncode}"
        )

    print()
    print(f"COMPLETED: {script_name}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    start_time = datetime.now()

    print("=" * 70)
    print("PENNSYLVANIA VOTER REGISTRATION DASHBOARD UPDATE")
    print("=" * 70)

    print(
        "Started:",
        start_time.strftime(
            "%Y-%m-%d %I:%M:%S %p"
        ),
    )

    for script_name in SCRIPTS:
        run_script(script_name)

    end_time = datetime.now()

    elapsed = (
        end_time
        - start_time
    ).total_seconds()

    print()
    print("=" * 70)
    print("DASHBOARD UPDATE COMPLETE")
    print("=" * 70)

    print(
        "Finished:",
        end_time.strftime(
            "%Y-%m-%d %I:%M:%S %p"
        ),
    )

    print(
        f"Elapsed time: {elapsed:.1f} seconds"
    )

    print()
    print(
        "Output:"
    )

    print(
        PROJECT_DIR
        / "docs"
        / "index.html"
    )


if __name__ == "__main__":
    main()
