"""Compatibility entry point: python run_batch.py --manifest ... --output ..."""
import sys
from material_transfer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["batch", *sys.argv[1:]]))
