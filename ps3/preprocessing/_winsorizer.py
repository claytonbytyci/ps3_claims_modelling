import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Parameters
        ----------
        lower_quantile : float
            Quantile for the lower bound (between 0 and 1).
        upper_quantile : float
            Quantile for the upper bound (between 0 and 1).
        """
        # __init__ just REMEMBERS these settings.
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Learn the clipping thresholds from the data.
        """
        X = np.asarray(X)

        # Compute column-wise quantiles and store them as learned attributes
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)

        return self  # sklearn convention

    def transform(self, X):
        """
        Apply winsorization: clip values to the learned quantile bounds.
        """
        # Make sure fit() has been called
        check_is_fitted(self, ["lower_quantile_", "upper_quantile_"])

        X = np.asarray(X)

        # np.clip will broadcast the per-column bounds
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)

        return X_clipped
