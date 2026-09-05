import pandas as pd

# PRINTING


def print_section(title):
    """
    Prints a formatted section header.
    """

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# Checkpoint Utilities


def load_checkpoint(checkpoint_file):
    """
    Loads checkpoint if it exists.

    Returns:
    checkpoint: DataFrame or None
    completed: set
    records: list
    """

    if checkpoint_file.exists():
        checkpoint = pd.read_csv(checkpoint_file)

        completed = set(checkpoint["imdb_id"])

        records = checkpoint.to_dict("records")

        return (
            checkpoint,
            completed,
            records,
        )

    return (
        None,
        set(),
        [],
    )


def save_checkpoint(
    records,
    checkpoint_file,
):
    """
    Saves checkpoint
    """

    checkpoint = pd.DataFrame(records)

    checkpoint.to_csv(
        checkpoint_file,
        index=False,
    )


# CSV Utilities


def save_dataframe(
    dataframe,
    path,
):
    """
    Saves dataframe to CSV.
    """

    dataframe.to_csv(
        path,
        index=False,
    )
