import argparse

from .config import CSV_PATH, PUBLIC_DATASET_SLUG, RAW_DATA_DIR, SPLIT_DIR
from .dataset_utils import (
    DatasetPreparationResult,
    build_records_from_image_folders,
    create_stratified_splits,
    download_kaggle_dataset,
    encode_labels,
    extract_valid_alzheimer_records_from_csv,
    save_splits,
)


def prepare_dataset(force_public_dataset: bool = False) -> DatasetPreparationResult:
    data = None
    source = "public_dataset"
    if not force_public_dataset:
        data = extract_valid_alzheimer_records_from_csv(CSV_PATH)
        if data is not None:
            source = "provided_csv"
    if data is None:
        download_kaggle_dataset(PUBLIC_DATASET_SLUG, RAW_DATA_DIR)
        data = build_records_from_image_folders(RAW_DATA_DIR)
    train_df, val_df, test_df = create_stratified_splits(data)
    train_df, val_df, test_df, class_to_index = encode_labels(train_df, val_df, test_df)
    train_csv, val_csv, test_csv = save_splits(train_df, val_df, test_df, class_to_index, SPLIT_DIR)
    return DatasetPreparationResult(
        source=source,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        class_to_index=class_to_index,
    )


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Alzheimer MRI dataset.")
    parser.add_argument("--force-public-dataset", action="store_true")
    args = parser.parse_args()
    result = prepare_dataset(force_public_dataset=args.force_public_dataset)
    print(f"Prepared dataset from: {result.source}")
    print(f"Train split: {result.train_csv}")
    print(f"Val split: {result.val_csv}")
    print(f"Test split: {result.test_csv}")
    print(f"Classes: {result.class_to_index}")


if __name__ == "__main__":
    main()

