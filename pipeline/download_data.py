"""Download the MovieLens Small dataset."""

import io
import os
import zipfile

import requests

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    dest = os.path.join(DATA_DIR, "ml-latest-small")
    if os.path.isdir(dest):
        print(f"Dataset already exists at {dest}, skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading MovieLens Small dataset from {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, timeout=120)
    resp.raise_for_status()

    print("Extracting ...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)

    print(f"Dataset extracted to {dest}")


if __name__ == "__main__":
    main()
