"""Download the Amazon Reviews 2018 Video Games 5-core and item metadata.

Source: https://nijianmo.github.io/amazon/index.html
Install dependencies from requirements.txt. Run with --help for options.
"""

import argparse
import gzip
import json
from pathlib import Path
import sys

import pooch


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
FILES = {
    "Video_Games_5.json.gz": (
        f"{BASE_URL}/categoryFilesSmall/Video_Games_5.json.gz",
        153_969_506,
    ),
    "meta_Video_Games.json.gz": (
        f"{BASE_URL}/metaFiles2/meta_Video_Games.json.gz",
        53_279_669,
    ),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def validate_archive(path, expected_size):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected size for {path.name}; expected {expected_size} bytes")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        if not isinstance(json.loads(source.readline()), dict):
            raise ValueError(f"Expected a JSON object in {path.name}")
        while source.read(1024 * 1024):
            pass


def download(url, destination, expected_size):
    path = Path(
        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=destination.name,
            path=destination.parent,
            downloader=pooch.HTTPDownloader(progressbar=True, timeout=60),
        )
    )
    validate_archive(path, expected_size)
    print(f"Archive checked: {path.name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def download_files(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_size) in FILES.items():
        destination = output_dir / filename
        download(url, destination, expected_size)


def main():
    args = parse_arguments()
    download_files(args.output_dir)
    print(f"Data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, EOFError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)
