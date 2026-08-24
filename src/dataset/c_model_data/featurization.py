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
    / "general_training.csv"
)

GENERAL_TEST_FEATURES = (
    GENERAL_DIR
    / "general_test.csv"
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

# FEATURIZE ONE DATASET 

def featurize_dataset(
    training_path,
    test_path,
    training_output,
    test_output,
    task,
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

    # Transform training 

    training_features = (
        preprocessor.transform(
            training_input
        )
    )

    # Transform test using SAME preprocessor 

    test_features = (
        preprocessor.transform(
            test_input
        )
    )

    training_features.insert(0, "imdb_id", training["imdb_id"].values)

    test_features.insert(0, "imdb_id", test["imdb_id"].values)
    
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
        / f"{task}_preprocessor.joblib",
    )

    # Print Result 

    print(
        f"{task}"
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
        task="general",
    )

    # CONTENT RATING

    featurize_dataset(
        training_path=CONTENT_RATING_TRAIN,
        test_path=CONTENT_RATING_TEST,
        training_output=CONTENT_RATING_TRAIN_FEATURES,
        test_output=CONTENT_RATING_TEST_FEATURES,
        task="content_rating",
    )

    # BOX OFFICE 

    featurize_dataset(
        training_path=BOX_OFFICE_TRAIN,
        test_path=BOX_OFFICE_TEST,
        training_output=BOX_OFFICE_TRAIN_FEATURES,
        test_output=BOX_OFFICE_TEST_FEATURES,
        task="box_office",
    )

    print(
        "\nFeature generation complete."
    )

if __name__ == "__main__": 

    main()