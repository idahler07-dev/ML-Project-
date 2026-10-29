"""Run the whole project with one command.

    python run_all.py               download fresh data, then run every step
    python run_all.py --no-download reuse the latest files in data/raw/ (gives
                                    exactly the results committed to the repository)

Steps: download -> clean -> features -> models -> evaluate -> plots.
If any step fails, the script stops with an error. No data is ever made up.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import clean  # noqa: E402
import download  # noqa: E402
import evaluate  # noqa: E402
import features  # noqa: E402
import models  # noqa: E402
import plots  # noqa: E402

SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-download", action="store_true",
                        help="use the latest raw files already in data/raw/")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    steps = [("download", download.main), ("clean", clean.main),
             ("features", features.main), ("models", models.main),
             ("evaluate", evaluate.main), ("plots", plots.main)]
    if args.no_download:
        steps = steps[1:]
        print("Skipping download: using the latest files in data/raw/.")

    for number, (name, step) in enumerate(steps, start=1):
        print(f"\n[{number}/{len(steps)}] {name}")
        start = time.time()
        step()
        print(f"  done in {time.time() - start:.1f} s")
    print("\nAll steps finished. Tables: results/tables/  Figures: results/figures/")


if __name__ == "__main__":
    main()
