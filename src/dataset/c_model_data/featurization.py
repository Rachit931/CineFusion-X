import pandas as pd 
from pathlib import Path 

from src.dataset.c_model_data.preprocessing import (
    TabularPreprocessor,
    save_preprocessor,
)

from config.paths import ( 
    S_GENERAL_DIR,
    S_CONTENT_RATING_DIR,
    S_BOX_OFFICE_SPLIT_DIR,
    GENERAL_DIR, 
    CONTENT_RATING_DIR,
    BOX_OFFICE_DIR,
    PREPROCESSOR_DIR,
)

import src.utils as utils 

# DATASET CONFIGURATION

# DATASET FILES

GENERAL_TRAIN = (
    S_GENERAL_DIR
    / "master_development.csv"
)

GENERAL_TEST = (
    S_GENERAL_DIR
    / "master_test.csv"
)

CONTENT_RATING_TRAIN = (
    S_CONTENT_RATING_DIR
    / "content_rating_development.csv"
)

CONTENT_RATING_TEST = (
    S_CONTENT_RATING_DIR
    / "content_rating_test.csv"
)

BOX_OFFICE_TRAIN = (
    S_BOX_OFFICE_SPLIT_DIR
    / "box_office_development.csv"
)

BOX_OFFICE_TEST = (
    S_BOX_OFFICE_SPLIT_DIR
    / "box_office_test.csv"
)


# FEATURE OUTPUT FILES

GENERAL_TRAIN_FEATURES = (
    GENERAL_DIR
    / "master_training.csv"
)

GENERAL_TEST_FEATURES = (
    GENERAL_DIR
    / "master_test.csv"
)

CONTENT_RATING_TRAIN_FEATURES = (
    CONTENT_RATING_DIR
    / "content_rating_training.csv"
)

CONTENT_RATING_TEST_FEATURES = (
    CONTENT_RATING_DIR
    / "content_rating_test.csv"
)

BOX_OFFICE_TRAIN_FEATURES = (
    BOX_OFFICE_DIR
    / "box_office_training.csv"
)

BOX_OFFICE_TEST_FEATURES = (
    BOX_OFFICE_DIR
    / "box_office_test.csv"
)

# RAW TABULAR FEATURES 

FEATURE_COLUMNS = [
    "budget",
    "runtime",
    "release_year",
    "release_date",
    "original_language",
    "production_countries",
    "spoken_languages",
    "production_companies",
]

TEXT_COLUMN = "overview"

BASE_TARGET_COLUMNS = [
    "rating_target",
    "box_office_target",
    "content_rating_target",
]

# FEATURIZE ONE DATASET 

def featurize_dataset(
    training_path,
    test_path,
    training_output,
    test_output,
): 
    """
    Fit the preprocessor on training data only,
    transform training and test data, 
    and save both processed DataFrames.
    """

    # Load already-split datasets 

    training = pd.read_csv(
        training_path,
        low_memory=False,
    )

    test = pd.read_csv(
        test_path,
        low_memory=False,
    )

    # Fidn genre target columns 

    genre_target_columns = [
        column 
        for column in training.columns
        if column.startswith("genre_")
        and column.endswith("_target")
    ]

    if len(genre_target_columns) != 19: 
        raise ValueError(
            f"Expected 19 genre target columns, "
            f"found{len(genre_target_columns)}"
        )

    # ALL target columns 

    target_columnns = [
        *genre_target_columns,
        *BASE_TARGET_COLUMNS
    ]

    # Select only the raw tabular features

    training_input = training[
        FEATURE_COLUMNS
    ].copy()

    test_input = test[
        FEATURE_COLUMNS
    ].copy()

    # Create preprocessor 

    preprocessor = TabularPreprocessor()

    # FIT ON TRAINING ONLY 

    preprocessor.fit(training_input)

    # TRANSFORM TRAINING

    training_features = (
        preprocessor.transform(
            training_input
        )
    )

    # Adding the targets 
    training_features.insert(0, "imdb_id", training["imdb_id"].values)

    # Add targets 
    for column in target_columnns:
        training_features[column] = training[column].values

    # Add overview
    training_features[TEXT_COLUMN] = training[TEXT_COLUMN].values


    # TRANSFORM TEST USING THE SAME PREPROCESSOR

    test_features = (
        preprocessor.transform(
            test_input
        )
    )

    # Adding the targets for model evaluation
    test_features.insert(0, "imdb_id", test["imdb_id"].values)

    # Add targets for evaluation
    for column in target_columnns:
        test_features[column] = test[column].values

    # Adding overview 
    test_features[TEXT_COLUMN] = test[TEXT_COLUMN].values

        
    # Save processed training DataFrame

    training_features.to_csv(
        training_output,
        index=False,
    )

    # Save processed testing DataFrame

    test_features.to_csv(
        test_output,
        index=False,
    )

    # Save fitted preprocessor 

    save_preprocessor(
        preprocessor,
        PREPROCESSOR_DIR 
        / "preprocessor.joblib",
    )

    # Print Result 

    print(
        f"training = {training_features.shape}, "
        f"test = {test_features.shape}"
    )

# MAIN 

def main(): 

    utils.print_section("FINAL FEATURE GENERATION")

    # GENERAL 

    featurize_dataset(
        training_path=GENERAL_TRAIN,
        test_path=GENERAL_TEST,
        training_output=GENERAL_TRAIN_FEATURES,
        test_output=GENERAL_TEST_FEATURES,
    )

    print(
        "\nFeature generation complete."
    )

if __name__ == "__main__": 

    main()