"""Clean invalid Amazon Video Games records and save the cleaned tables."""

import argparse
import html
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "__unknown__"
REVIEW_COLUMNS = {
    "reviewerID": "user_id",
    "asin": "item_id",
    "overall": "rating",
    "unixReviewTime": "timestamp",
}


def read_columns(path, columns):
    with pd.read_json(path, lines=True, chunksize=20_000, dtype=False) as reader:
        chunks = [chunk.reindex(columns=columns) for chunk in reader]
    return pd.concat(chunks, ignore_index=True)


def remove_invalid_reviews(raw):
    data = raw.rename(columns=REVIEW_COLUMNS).copy()
    for column in ["user_id", "item_id"]:
        data[column] = data[column].astype("string").str.strip().replace("", pd.NA)

    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    seconds = pd.to_numeric(data["timestamp"], errors="coerce")
    data["timestamp"] = pd.to_datetime(seconds, unit="s", errors="coerce", utc=True)
    valid = (
        data[["user_id", "item_id", "timestamp"]].notna().all(axis=1)
        & data["rating"].isin([1, 2, 3, 4, 5])
        & seconds.ge(0)
    )
    return data.loc[valid].copy(), int((~valid).sum())


def deduplicate_reviews(data):
    unique = data.drop_duplicates()
    removed = {
        "duplicate_interactions_removed": len(data) - len(unique),
    }
    return unique, removed


def clean_reviews(raw):
    valid, invalid_count = remove_invalid_reviews(raw)
    data, duplicate_counts = deduplicate_reviews(valid)
    data["rating"] = data["rating"].astype("int8")
    report = {
        "raw_rows": len(raw),
        "invalid_rows_removed": invalid_count,
        **duplicate_counts,
    }
    return data.reset_index(drop=True), report


def prepare_interactions(path):
    raw = read_columns(path, list(REVIEW_COLUMNS))
    return clean_reviews(raw)


def clean_text(value):
    if not isinstance(value, str):
        return ""
    value = re.sub(r"<[^>]*>", " ", html.unescape(value))
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value.lower() in {"none", "null", "nan", "n/a"} else value


def clean_categories(value):
    if not isinstance(value, (list, tuple)):
        value = [value] if isinstance(value, str) else []
    labels = [clean_text(entry) for entry in value]
    return tuple(sorted({label for label in labels if label}))


def normalize_item_metadata(raw, item_ids):
    data = raw.rename(columns={"asin": "item_id", "category": "categories"}).copy()
    data["item_id"] = data["item_id"].astype("string").str.strip()
    data = data[data["item_id"].isin(item_ids)].copy()
    for column in ["title", "brand"]:
        data[column] = data[column].map(clean_text)
    data["categories"] = data["categories"].map(clean_categories)
    return data


def select_item_records(data):
    unique = data.drop_duplicates(["item_id", "title", "brand", "categories"]).copy()
    unique["_filled"] = (
        unique["title"].ne("").astype(int)
        + unique["brand"].ne("").astype(int)
        + unique["categories"].map(bool).astype(int)
    )
    unique["_category_count"] = unique["categories"].map(len)
    selected = unique.sort_values(
        ["item_id", "_filled", "_category_count"],
        ascending=[True, False, False], kind="stable",
    ).drop_duplicates("item_id", keep="first")
    removed = {
        "duplicate_metadata_rows_removed": len(data) - len(unique),
        "alternative_metadata_rows_removed": len(unique) - len(selected),
    }
    return selected.drop(columns=["_filled", "_category_count"]), removed


def fill_missing_item_metadata(metadata, item_ids):
    catalog = pd.DataFrame({"item_id": sorted(item_ids)})
    catalog = catalog.merge(
        metadata, on="item_id", how="left", validate="one_to_one", indicator=True,
    )
    catalog["metadata_missing"] = catalog.pop("_merge").eq("left_only")
    catalog["brand"] = catalog["brand"].fillna("").replace("", UNKNOWN)
    catalog["title"] = catalog["title"].fillna("")
    catalog.loc[catalog["title"].eq(""), "title"] = catalog["item_id"]
    catalog["categories"] = catalog["categories"].map(
        lambda value: list(value) if isinstance(value, tuple) and value else [UNKNOWN]
    )
    return catalog


def clean_items(raw, item_ids):
    normalized = normalize_item_metadata(raw, item_ids)
    selected, duplicate_counts = select_item_records(normalized)
    catalog = fill_missing_item_metadata(selected, item_ids)
    report = {
        "raw_metadata_rows": len(raw),
        "metadata_rows_for_selected_items": len(normalized),
        **duplicate_counts,
        "items_without_metadata": int(catalog["metadata_missing"].sum()),
        "items_with_unknown_brand": int(catalog["brand"].eq(UNKNOWN).sum()),
        "items_with_unknown_categories": int(
            catalog["categories"].map(lambda value: value == [UNKNOWN]).sum()
        ),
    }
    return catalog, report


def prepare_items(path, interactions):
    raw = read_columns(path, ["asin", "title", "brand", "category"])
    return clean_items(raw, set(interactions["item_id"]))


def summarize_reviews(data):
    users = data["user_id"].nunique()
    items = data["item_id"].nunique()
    return {
        "rows": len(data),
        "users": int(users),
        "items": int(items),
        "first_timestamp": data["timestamp"].min().isoformat() if len(data) else None,
        "last_timestamp": data["timestamp"].max().isoformat() if len(data) else None,
        "unique_user_item_pairs": len(data.drop_duplicates(["user_id", "item_id"])),
        "rating_counts": {
            str(key): int(value)
            for key, value in data["rating"].value_counts().sort_index().items()
        },
    }


def validate_metadata_join(interactions, items):
    joined = interactions.merge(
        items[["item_id"]], on="item_id", validate="many_to_one",
    )
    if len(joined) != len(interactions):
        raise ValueError("Joining metadata changed the interaction count")


def build_preparation_report(interactions, review_report, metadata_report):
    return {
        **review_report,
        "metadata": metadata_report,
        "cleaned": summarize_reviews(interactions)
    }


def save_prepared_data(output_dir, interactions, items, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"interactions": interactions, "items": items}
    for name, frame in tables.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Reading and cleaning reviews...", flush=True)
    interactions, review_report = prepare_interactions(
        args.raw_dir / "Video_Games_5.json.gz",
    )

    print("Reading and cleaning item metadata...", flush=True)
    items, metadata_report = prepare_items(
        args.raw_dir / "meta_Video_Games.json.gz", interactions,
    )
    validate_metadata_join(interactions, items)

    report = build_preparation_report(
        interactions, review_report, metadata_report,
    )
    save_prepared_data(args.output_dir, interactions, items, report)

    print(json.dumps(report["cleaned"], ensure_ascii=False, indent=2))
    print(f"Cleaned data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
