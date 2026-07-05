import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from configs.config import CSV_DIR, JPEG_DIR, SEED


def load_cbis_csv_files(csv_dir=CSV_DIR):
    mass_train = pd.read_csv(csv_dir / "mass_case_description_train_set.csv")
    mass_test = pd.read_csv(csv_dir / "mass_case_description_test_set.csv")

    calc_train = pd.read_csv(csv_dir / "calc_case_description_train_set.csv")
    calc_test = pd.read_csv(csv_dir / "calc_case_description_test_set.csv")

    print("Mass train:", mass_train.shape)
    print("Mass test:", mass_test.shape)
    print("Calc train:", calc_train.shape)
    print("Calc test:", calc_test.shape)

    return mass_train, mass_test, calc_train, calc_test


def prepare_cbis_dataframe(df, lesion_type, split):
    df = df.copy()
    df["lesion_type"] = lesion_type
    df["split"] = split

    if lesion_type == "calcification":
        df["lesion_label"] = 0
        df["lesion_name"] = "Calcification"
    else:
        df["lesion_label"] = 1
        df["lesion_name"] = "Mass"

    df["pathology_label"] = df["pathology"].map({
        "BENIGN": 0,
        "BENIGN_WITHOUT_CALLBACK": 0,
        "MALIGNANT": 1,
    })

    df = df.dropna(subset=["pathology_label"]).reset_index(drop=True)
    df["pathology_label"] = df["pathology_label"].astype(int)

    keep_cols = [
        "patient_id",
        "left or right breast",
        "image view",
        "abnormality id",
        "abnormality type",
        "pathology",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
        "lesion_type",
        "lesion_label",
        "lesion_name",
        "pathology_label",
        "split",
    ]

    return df[keep_cols]


def build_full_dataframes(
    mass_train,
    mass_test,
    calc_train,
    calc_test,
    seed=SEED,
):
    calc_train_df = prepare_cbis_dataframe(calc_train, "calcification", "train")
    calc_test_df = prepare_cbis_dataframe(calc_test, "calcification", "test")

    mass_train_df = prepare_cbis_dataframe(mass_train, "mass", "train")
    mass_test_df = prepare_cbis_dataframe(mass_test, "mass", "test")

    full_train_df = pd.concat(
        [calc_train_df, mass_train_df],
        ignore_index=True,
    )

    test_df = pd.concat(
        [calc_test_df, mass_test_df],
        ignore_index=True,
    )

    full_train_df = full_train_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    print("Full train:", full_train_df.shape)
    print("Test:", test_df.shape)

    print("\nLesion distribution:")
    print(full_train_df["lesion_name"].value_counts())

    print("\nPathology distribution:")
    print(full_train_df["pathology_label"].value_counts())

    return full_train_df, test_df


def extract_uid_from_csv_path(csv_path):
    parts = str(csv_path).split("/")

    if len(parts) >= 3:
        return parts[2]

    return None


def build_uid_to_image_path(jpeg_dir=JPEG_DIR):
    uid_to_path = {}

    for folder in tqdm(list(jpeg_dir.iterdir()), desc="Indexing images"):
        if folder.is_dir():
            image_files = sorted(
                list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
                + list(folder.glob("*.png"))
            )

            if len(image_files) > 0:
                uid_to_path[folder.name] = image_files[0]

    return uid_to_path


def resolve_image_path(csv_path, uid_to_path):
    uid = extract_uid_from_csv_path(csv_path)

    if uid is None:
        return None

    return uid_to_path.get(uid, None)


def remove_missing_rows(df, columns, uid_to_path):
    valid_indices = []

    for idx, row in df.iterrows():
        is_valid = True

        for col in columns:
            if resolve_image_path(row[col], uid_to_path) is None:
                is_valid = False
                break

        if is_valid:
            valid_indices.append(idx)

    return df.loc[valid_indices].reset_index(drop=True)


def split_train_validation(full_train_df, test_df, seed=SEED):
    full_train_df["stratify_label"] = (
        full_train_df["lesion_label"].astype(str)
        + "_"
        + full_train_df["pathology_label"].astype(str)
    )

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.15,
        stratify=full_train_df["stratify_label"],
        random_state=seed,
    )

    train_df = train_df.drop(columns=["stratify_label"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["stratify_label"]).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nTrain lesion distribution:")
    print(train_df["lesion_name"].value_counts())

    print("\nTrain pathology distribution:")
    print(train_df["pathology_label"].value_counts())

    return train_df, val_df, test_df


def prepare_data():
    print("CSV_DIR:", CSV_DIR.exists())
    print("JPEG_DIR:", JPEG_DIR.exists())

    if CSV_DIR.exists():
        print(os.listdir(CSV_DIR))

    mass_train, mass_test, calc_train, calc_test = load_cbis_csv_files(CSV_DIR)

    full_train_df, test_df = build_full_dataframes(
        mass_train,
        mass_test,
        calc_train,
        calc_test,
    )

    uid_to_path = build_uid_to_image_path(JPEG_DIR)

    print("Indexed image folders:", len(uid_to_path))

    full_train_df = remove_missing_rows(
        full_train_df,
        ["cropped image file path"],
        uid_to_path,
    )

    test_df = remove_missing_rows(
        test_df,
        ["cropped image file path"],
        uid_to_path,
    )

    print("After removing missing rows")
    print("Full train:", full_train_df.shape)
    print("Test:", test_df.shape)

    train_df, val_df, test_df = split_train_validation(
        full_train_df,
        test_df,
    )

    return train_df, val_df, test_df, uid_to_path
