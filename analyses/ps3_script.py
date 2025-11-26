# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# load data
df = load_transform()

# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# We divide by exposure to model pure premium (expected claims per unit exposure).

# Split data into train/test sets using deterministic hash of the policy ID
df = create_sample_split(df, id_column="IDpol", training_frac=0.7)
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)

# Add splines for BonusMalus and Density and use a Pipeline.
# Steps:
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer.
#    Choose knots="quantile" for the SplineTransformer and make sure, we
#    are only including one intercept in the final GLM.
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_cols = ["BonusMalus", "Density"]

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "spline",
            SplineTransformer(
                degree=3,
                n_knots=5,
                knots="quantile",
                include_bias=False,  # <- no extra intercept here
            ),
        ),
    ]
)

# Let's put together a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
    ]
)

preprocessor.set_output(transform="pandas")

glm_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "estimate",
            GeneralizedLinearRegressor(
                family=TweedieDist, l1_ratio=1, fit_intercept=True
            ),
        ),
    ]
)


# let's have a look at the pipeline
glm_pipeline

# let's check that the transforms worked
glm_pipeline[:-1].fit_transform(df_train)

glm_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([glm_pipeline[-1].intercept_], glm_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + glm_pipeline[-1].feature_names_,
).T

df_test["pp_t_glm2"] = glm_pipeline.predict(df_test)
df_train["pp_t_glm2"] = glm_pipeline.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# Use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.

# 1: Define the modelling pipeline (no preprocessing needed, X_train_t is already numeric)
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

lgbm_fast = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    learning_rate=0.1,
    n_estimators=80,
    subsample=0.5,
    colsample_bytree=0.5,
    num_leaves=12,
    max_depth=4,
    n_jobs=-1,
    force_col_wise=True,
    random_state=0,
)

lgbm_pipeline = Pipeline([("estimate", lgbm_fast)])

lgbm_pipeline.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
)

df_test["pp_t_lgbm"] = lgbm_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = lgbm_pipeline.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# Tune the LGBM to reduce overfitting with a small grid search.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param.

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
#%%
param_grid = {
    "estimate__learning_rate": [0.01, 0.05, 0.1],
    "estimate__n_estimators": [200, 500, 800],
}

cv = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=param_grid,
    cv=5,
)
#%%
cv.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)

# Let's compare the sorting of the pure premium predictions

# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
## PS4

# Ex 1 Monotonicity constraints
## **Objective**
# We consider again the risk modelling task of problem set 3. The bonus malus feature tracks the
# claim history of the customer and the price is expected to increase if their bonus malus score
# gets worse and decrease if their score improves. We will likely find this to hold empirically
#  in a simple model with enough exposure for each bin, but the more complex our models get,
#  i.e. the more interactions they entail, the more likely it is that there might be edge cases
#  in which this monotonicity breaks. Hence, we would like to include an explicit monotonicity
# constraint for this feature.

## Tasks
# Train a constrained LGBM by introducing a monotonicity constraint for `BonusMalus` into the
# `LGBMRegressor`. Cross-validate is and compare the prediction accuracy of the constrained
# estimator to the unconstrained one.

# 1. Create a plot of the average claims per BonusMalus group,
# make sure to weigh them by exposure.
# What will/could happen if we do not include a monotonicity constraint?
df_bonus = df.groupby("BonusMalus").agg(
    total_claims=("ClaimAmountCut", "sum"),
    total_exposure=("Exposure", "sum")
).reset_index()
df_bonus["avg_claims_per_bonusmalus"] = df_bonus["total_claims"] / df_bonus["total_exposure"]
plt.figure(figsize=(10,6))
plt.plot(df_bonus["BonusMalus"], df_bonus["avg_claims_per_bonusmalus"], marker='o')
plt.title("Average Claims per BonusMalus Group (Weighted by Exposure)")
plt.xlabel("BonusMalus")
plt.ylabel("Average Claims per Unit Exposure")
plt.grid()
plt.show()
# Without a monotonicity constraint, the model might learn a non-monotonic relationship
# between BonusMalus and expected claims, which could lead to counterintuitive pricing
# where better drivers (lower BonusMalus) are predicted to have higher claims than worse drivers
# %%
# 2. Create a new model pipeline or estimator called constrained_lgbm.
# Introduce an increasing monotonicity constrained for BonusMalus.
# Note: We have to provide a list of the same length as our features with 0s
# everywhere except for BonusMalus where we put a 1.
# See: Parameters — LightGBM 4.5.0.99 documentation
monotonicity_constraints = []
for col in X_train_t.columns:
    if col == "BonusMalus":
        monotonicity_constraints.append(1)
    else:
        monotonicity_constraints.append(0)

constrained_lgbm = LGBMRegressor(
    objective="tweedie",
    tweedie_variance_power=1.5,
    learning_rate=0.2,
    n_estimators=80,
    subsample=0.5,
    colsample_bytree=0.5,
    num_leaves=12,
    max_depth=4,
    n_jobs=-1,
    force_col_wise=True,
    random_state=0,
    monotone_constraints=monotonicity_constraints
)
constrained_lgbm_pipeline = Pipeline([("estimate", constrained_lgbm)])
#3. Cross-validate and predict using the best estimator.
# Save the predictions in the column pp_t_lgbm_constrained.

param_grid = {
    "estimate__learning_rate": [0.04, 0.05, 0.1],
    "estimate__n_estimators": [40, 80, 100],
}

cv_constrained = GridSearchCV(
    estimator=constrained_lgbm_pipeline,
    param_grid=param_grid,
    cv=5,
)
cv_constrained.fit(X_train_t, y_train_t, estimate__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)
print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

#%%
# Ex2 Learning Curve
## **Objective**
# For iterative optimisation algorithms, such as GBMs, it is important to track their
# convergence in order to understand, whether they converged successfully or by what extend
#  they are under- or overfitting.

# Task
# Based on the cross-validated constrained `LGBMRegressor` object,
# plot a learning curve which is showing the convergence of the score on the train and test set.

# Steps

# 1. Re-fit the best constrained lgbm estimator from the cross-validation and
# provide the tuples of the test and train dataset to the estimator via `eval_set`.

cv_constrained.best_estimator_.fit(
    X_train_t,
    y_train_t,
    estimate__sample_weight=w_train_t,
    estimate__eval_set=[(X_train_t, y_train_t), (X_test_t, y_test_t)],
    estimate__eval_sample_weight=[w_train_t, w_test_t],
    estimate__eval_metric="tweedie",
    estimate__verbose=False,
)

# 2. Plot the learning curve by running `lgb.plot_metric` on the
# estimator (either the estimator directly or as last step of the pipeline).

import lightgbm as lgb
lgb.plot_metric(cv_constrained.best_estimator_.named_steps["estimate"])
plt.title("Learning Curve for Constrained LGBM")
plt.show()

# 3. What do you notice, is the estimator tuned optimally?

# The learning curve shows that the training loss continues to decrease,
# indicating that the model is still learning from the training data.
# However, the validation loss starts to increase after a certain number of iterations,
# suggesting that the model begins to overfit the training data.
# This indicates that the estimator is not tuned optimally,
# and further tuning or early stopping may be necessary to achieve better generalization.
# %%
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
# 3. Compute the bias of your estimates as deviation
#  from the actual exposure adjusted mean
# 4. Compute the deviance.
# 5. Compute the MAE and RMSE.

from ps3_claims_modelling.evaluation._evaluate_predictions import evaluate_predictions

metrics_constrained = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm_constrained"],
    exposure=w_test_t
)
metrics_unconstrained = evaluate_predictions(
    y_true=y_test_t,
    y_pred=df_test["pp_t_lgbm"],
    exposure=w_test_t
)

print("Metrics for Constrained LGBM:")
print(metrics_constrained)
print("\nMetrics for Unconstrained LGBM:")
print(metrics_unconstrained)

# 6. Bonus: Compute the Gini coefficient as defined in the plot
#  of the Lorentz curve at the bottom of `ps3_script`.

lorenz_x, lorenz_y = lorenz_curve(
    y_test_t,
    df_test["pp_t_lgbm_constrained"],
    w_test_t
)
gini_constrained = 1 - 2 * auc(lorenz_x, lorenz_y)

lorenz_x, lorenz_y = lorenz_curve(
    y_test_t,
    df_test["pp_t_lgbm"],
    w_test_t
)
gini_unconstrained = 1 - 2 * auc(lorenz_x, lorenz_y)

print(f"Gini Coefficient for Constrained LGBM: {gini_constrained:.4f}")
print(f"Gini Coefficient for Unconstrained LGBM: {gini_unconstrained:.4f}")

# 7. Return a dataframe with the names of the metrics as index.

metrics_df = pd.DataFrame({
    "Constrained LGBM": metrics_constrained,
    "Unconstrained LGBM": metrics_unconstrained
})

# 8. Use the function and compare the constrained
# and unconstrained lgbm models.

print("\nComparison of Metrics:")
print(metrics_df)

#%%
# # Ex 4 Evaluation plots

## **Objective**
# Now that we have fitted several models, and computed some metrics,
# we want to better understand how specific features
# drive predictions. One model-agnostic way to illustrate the marginal
# effects are Partial Dependence Plots (PDPs).
# See: https://christophm.github.io/interpretable-ml-book/pdp.html

### Task
# Plots the PDPs of all features and compare the PDPs between the
# unconstrained and constrained LGBM.
# Use the DALEX package as it offers more functionality as the
# `sklearn.inspection` module.

# Steps:
# 1. Define an explainer object using the constrained lgbm model, data and features.
# For help, see here: https://ema.drwhy.ai/partialDependenceProfiles.html#PDPPython

# 2. Now compute the marginal effects using `model_profile`
# and plot the PDPs.
# Note: If you receive a `numpy` error related to `ptp` simply downgrade `numpy` to <2 by `conda install numpy<2` and restarting your environment. The DALEX package does only support `numpy>=2` for versions >1.7.0. You might also have to install `nbformat`.

# Ex 5 Shapley

## **Objective**
# Sometimes we want to understand model predictions on a more granular
# level, particularly, what features are driving predictions
# upwards or downwards and by how much. This interpretation demands
# additivity in the feature effects and this is exactly what SHAP
# (SHapley Additive exPlanations) values (https://arxiv.org/abs/1705.07874) are providing us.
# Task
# Compare the decompositions of the predictions for some specific row
# (e.g. the first row of the test set) for the constrained LGBM
# and our initial GLM.

# Step:
# 1. Define DALEX Explainer objects for both.

# 2. Call the method `predict_parts` for each and provide one
# observation as data point and `type="shap"`.

# 3. Plot both decompositions and compare where they might deviate.
