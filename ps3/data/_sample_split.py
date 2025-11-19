import hashlib

import numpy as np


# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation. #DONE.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if id_column not in df.columns:
        raise ValueError(f"{id_column} not found in dataframe columns.")

    if not (0 < training_frac < 1):
        raise ValueError("training_frac must be between 0 and 1.")

    # Use a stable hash based on the ID values converted to string
    id_as_str = df[id_column].astype(str)

    def _hash_to_bucket(x: str, n_buckets: int = 100) -> int:
        digest = hashlib.md5(x.encode("utf-8")).hexdigest()
        return int(digest, 16) % n_buckets

    buckets = id_as_str.apply(_hash_to_bucket)

    # Map buckets to train / test based on training_frac
    threshold = int(training_frac * 100)
    is_train = buckets < threshold

    df = df.copy()
    df["sample"] = np.where(is_train, "train", "test")

    return df
