from .merge import merge_csv_files

from .validation import validate_schema

from .cleaning import (
    clean_column_names,
    remove_duplicates,
    replace_infinity,
    convert_features_to_numeric,
    handle_missing_values,
    handle_outliers,
)

from .feature_selection import (
    remove_constant_quasi_constant_features,
    remove_highly_correlated_features,
)

from .encoding import encode_labels

from .scaling import scale_features

from .saving import save_dataset

from sklearn.model_selection import train_test_split


def main():

    merged_df = merge_csv_files()

    df = clean_column_names(merged_df)

    df = remove_duplicates(df)

    df = replace_infinity(df)

    df = convert_features_to_numeric(df)

    df = handle_missing_values(df)

    df = remove_constant_quasi_constant_features(df)

    df = remove_highly_correlated_features(df)

    df = handle_outliers(df)

    df = encode_labels(df)

    X = df.drop("Label", axis=1)

    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_test = scale_features(
        X_train,
        X_test
    )

    save_dataset(
        merged_df,
        df,
        X_train,
        X_test,
        y_train,
        y_test
    )