"""Invert bintoindex.json into indextobin.json for downstream lookups."""

import os
import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert bintoindex.json to indextobin.json by swapping keys and values.")
    parser.add_argument("folder", help="Path to the folder containing bintoindex.json")
    args = parser.parse_args()

    bintoindex_path = os.path.join(args.folder, "bintoindex.json")
    indextobin_path = os.path.join(args.folder, "indextobin.json")

    if not os.path.isfile(bintoindex_path):
        print(f"Error: {bintoindex_path} does not exist.")
        sys.exit(1)

    with open(bintoindex_path, 'r') as f:
        bintoindex = json.load(f)

    # Invert key-value pairs
    try:
        indextobin = {str(v): k for k, v in bintoindex.items()}
    except Exception as e:
        print(f"Error during inversion: {e}")
        sys.exit(1)

    with open(indextobin_path, 'w') as f:
        json.dump(indextobin, f, indent=4)

    print(f"Successfully wrote {indextobin_path}")

if __name__ == "__main__":
    main()
