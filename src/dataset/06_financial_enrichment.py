import os 
import re
from pathlib import Path
import pandas as pd 
import numpy as np
import src.utils as utils

from config.paths import (
    EXTERNAL_DIR,
    BASE_DIR,
    MASTER_MULTIMODEL_DIR
)

# PATHS 

BASE = BASE_DIR / "multimodel_dataset.csv"

ENRICHED = MASTER_MULTIMODEL_DIR / "multimodel_dataset_enriched.csv"

# COLUMN ALIASES 

ALIASES = {
    "imdb_id" : [
        "tconst", "titleId", "id", "imdb_id", 
        "imdbid", "imdb-id", "imdb id", "imdb",
        "imdbID", "imdbId", "tconst", "imdb_title_id", 
        "imdb_titleid"
    ],

    "budget": [
        "budget", "movie_budget", "production_budget",
        "productionbudget", "film_budget", "estimated_budget"
    ],

    "revenue": [
        "revenue", "movie_revenue", "box_office", "boxoffice",
        "box_office_revenue", "worldwide_revenue", "worldwide_gross",
        "gross_worldwide", "worldwide_box_office", "gross", "total_gross", 
        "total_revenue"
    ],
}


def find_column(columns, aliases):
    columns = {
        re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_"): col
        for col in columns
    }

    for alias in aliases:
        alias = re.sub(
            r"[^a-z0-9]+",
            "_",
            alias.lower()
        ).strip("_")

        if alias in columns:
            return columns[alias]

    return None

def load_external_file(file): 
    data = pd.read_csv(
        file,
        low_memory=False,
    )

    imdb_col = find_column(
        data.columns,
        ALIASES["imdb_id"]
    )

    budget_col = find_column(
        data.columns,
        ALIASES["budget"]
    )

    revenue_col = find_column(
        data.columns,
        ALIASES["revenue"]
    )

    if imdb_col is None: 
        raise ValueError(
            f"No IMDB ID column found in {file.name}"
        )

    if budget_col is None:
        raise ValueError(
            f"No budget column found in {file.name}"
        )

    if revenue_col is None:
        raise ValueError(
            f"No revenue column found in {file.name}"
        )

    data = data[
        [imdb_col, budget_col, revenue_col]
    ].copy()

    data.columns = [
        "imdb_id",
        "budget",
        "revenue"
    ]

    data["imdb_id"] = (
        data["imdb_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert financial columns to numbers
    data["budget"] = pd.to_numeric(
        data["budget"],
        errors="coerce"
    )

    data["revenue"] = pd.to_numeric(
        data["revenue"],
        errors="coerce"
    )

    # Zero / negative financial values are treated as missing
    data.loc[data["budget"] <= 0, "budget"] = np.nan
    data.loc[data["revenue"] <= 0, "revenue"] = np.nan

    return data


def main(): 

    df = pd.read_csv(
        BASE,
        low_memory=False,
    )

    df["imdb_id"] = (
        df["imdb_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    print("Original Shape: ", df.shape)

    files = sorted(
        EXTERNAL_DIR.glob("*.csv")
    )

    external_data = []

    for file in files: 

        print("Reading : ", file.name)

        data = load_external_file(file)

        external_data.append(data)

    external = pd.concat(
        external_data,
        ignore_index=True
    )

    financial = (
        external
        .groupby("imdb_id")[["budget", "revenue"]]
        .median()
    )

    df["budget"] = pd.to_numeric(
        df["budget"],
        errors="coerce"
    )

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    df.loc[df["budget"] <= 0, "budget"] = np.nan
    df.loc[df["revenue"] <= 0, "revenue"] = np.nan


    df = df.set_index("imdb_id")

    df["budget"] = (
        df["budget"]
        .fillna(financial["budget"])
    )

    df["revenue"] = (
        df["revenue"]
        .fillna(financial["revenue"])
    )

    df = df.reset_index()

    utils.save_dataframe(
        df,
        ENRICHED
    )

    utils.print_section(
        "FINANCIAL RECOVERY"
    )

    print("\nEnrichment complete.")
    print("Final shape:", df.shape)

    print(
        "Budget available:",
        df["budget"].notna().sum()
    )

    print(
        "Revenue available:",
        df["revenue"].notna().sum()
    )

    print(
        "Both available:",
        (
            df["budget"].notna()
            & df["revenue"].notna()
        ).sum()
    )

    print("\nSaved to:")
    print(ENRICHED)


if __name__ == "__main__":
    main()