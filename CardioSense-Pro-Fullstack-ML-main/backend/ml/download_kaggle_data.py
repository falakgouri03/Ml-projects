from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Heart Disease dataset from Kaggle")
    parser.add_argument(
        "--dataset-slug",
        default=os.getenv("KAGGLE_DATASET_SLUG", "fedesoriano/heart-failure-prediction"),
        help="Kaggle dataset slug in format owner/dataset-name",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw"),
        help="Directory where dataset files will be extracted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
        "--force",
    ]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Kaggle CLI not found. Install dependency and ensure kaggle command is available."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Kaggle download failed. Verify dataset slug and Kaggle API credentials (~/.kaggle/kaggle.json)."
        ) from exc

    print(f"Dataset downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
