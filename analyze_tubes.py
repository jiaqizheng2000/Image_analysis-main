"""Analyze a folder without the desktop interface, using a saved setup."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from tube_analysis import Setup, list_images, run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Folder containing images")
    parser.add_argument("--setup", type=Path, help="Saved setup JSON (default: INPUT/tube_setup.json)")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--no-previews", action="store_true", help="Skip annotated first/middle/last images")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/PDF height plots")
    args = parser.parse_args()
    try:
        files = list_images(args.input)
        setup = Setup.load(args.setup or args.input / "tube_setup.json")
        output = args.output or Path(__file__).resolve().parent / "results" / f"{args.input.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        result = run_batch(files, setup, output, progress=lambda done, total, name: print(f"[{done}/{total}] {name}", flush=True), save_previews=not args.no_previews, save_plots=not args.no_plots)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 2 if result["failed_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
