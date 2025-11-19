import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    w = Winsorizer(lower_quantile, upper_quantile)
    w.fit(X)

    # Check learned attributes
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)

    assert np.isclose(w.lower_quantile_, expected_lower)
    assert np.isclose(w.upper_quantile_, expected_upper)

    # Transform
    Xt = w.transform(X)

    # Check clipping was applied
    assert Xt.min() >= w.lower_quantile_
    assert Xt.max() <= w.upper_quantile_

    # Check shape preservation
    assert Xt.shape == X.shape
