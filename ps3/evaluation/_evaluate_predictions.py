# Ex 3 Metrics function
## **Objective**
# When running multiple models in various contexts, it is convenient
#  to have a function which returns various performance metrics
# of the model, so we don’t have to compute them every time from
# scratch again.

### Task
# Write a function `evaluate_predictions` within the `evaluation`
# module, which computes various metrics given the true outcome
# values and the model’s predictions.
# Steps:
# 1. Create a module folder `evaluation` and an empty `__init__.py`
# which we will use to register the function at the module level.
# 2. Create a new file `_evaluate_predictions.py` in which you create
#  the respective function which takes the predictions
# and actuals as input, as well as some sample weight
# (in our case exposure).

# The function should compute and return the following metrics:
# - Mean Absolute Error (MAE)
# - Root Mean Squared Error (RMSE)
# - Tweedie Deviance (with power parameter 1.5)
# - Bias (i.e. the difference between the mean of predictions
#   and the mean of actuals, weighted by exposure)


def evaluate_predictions(
    y_true,
    y_pred,
    sample_weight=None,
):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.metrics import mean_tweedie_deviance

    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    tweedie_deviance = mean_tweedie_deviance(
        y_true, y_pred, power=1.5, sample_weight=sample_weight
    )
    bias = np.average(y_pred, weights=sample_weight) - np.average(
        y_true, weights=sample_weight
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Tweedie Deviance": tweedie_deviance,
        "Bias": bias,
    }
