from typing import Dict, Any, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import LeaveOneOut, cross_val_score
import optuna
from optuna.samplers import GPSampler, TPESampler
from itertools import product
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from optuna.importance import get_param_importances
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor



# Helper function to determine if an estimator supports multi-output regression
def supports_multi_output(estimator: BaseEstimator) -> bool:
    """
    Determines if the given estimator supports multi-output regression natively.

    Args:
        estimator (BaseEstimator): The scikit-learn estimator to check.

    Returns:
        bool: True if the estimator supports multi-output regression, False otherwise.
    """
    # List of estimators known to support multi-output regression
    multi_output_estimators = (
        LinearRegression,
        RandomForestRegressor,
        XGBRegressor,
        MultiTaskLasso,
        # Add other estimators that support multi-output regression here
    )
    return isinstance(estimator, multi_output_estimators)


def create_optuna_param_distributions(params: Dict[str, Union[List[Any], Any]]) -> Dict[str, Any]:
    optuna_params = {}
    log_scale_params = ["min_impurity_decrease", "ccp_alpha", "alpha", "tol", "eta", "gamma"]
    for param, values in params.items():
        if not isinstance(values, list):
            values = [values]

        non_none_values = [v for v in values if v is not None]
        if all(isinstance(v, int) for v in non_none_values):
            optuna_params[param] = optuna.distributions.IntDistribution(
                low=min(non_none_values),
                high=max(non_none_values)
            )
        elif all(isinstance(v, (int, float)) for v in non_none_values):
            if param in log_scale_params:
                optuna_params[param] = optuna.distributions.FloatDistribution(
                    low=min(non_none_values),
                    high=max(non_none_values),
                    log=True
                )
            else:
                optuna_params[param] = optuna.distributions.FloatDistribution(
                    low=min(non_none_values),
                    high=max(non_none_values)
                )
        else:
            optuna_params[param] = optuna.distributions.CategoricalDistribution(choices=values)

    return optuna_params


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: SMAPE value.
    """
    # Add epsilon to denominator to avoid division by zero
    epsilon = 1e-8
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon),
        axis=0  # Compute along columns (outputs)
    ) * 100
    return np.mean(smape)  # Average over all outputs


def create_cv_object(meta: pd.Series, n_splits: int = 5) -> Tuple[Union[KFold, LeaveOneGroupOut], np.ndarray]:
    """
    Creates a cross-validation object based on the number of unique groups in meta.

    Args:
        meta (pd.Series): Metadata for grouping.
        n_splits (int, optional): Number of splits for KFold. Defaults to 5.

    Returns:
        Tuple[Union[KFold, LeaveOneGroupOut], np.ndarray]: Cross-validation object and encoded groups.
    """
    le = LabelEncoder()
    encoded_meta = le.fit_transform(meta)
    unique_groups = np.unique(encoded_meta)
    n_groups = len(unique_groups)

    print(f"[create_cv_object] Number of unique groups: {n_groups}")

    if n_groups == 1:
        print(f"[create_cv_object] Only one group detected. Using {n_splits}-fold KFold.")
        return KFold(n_splits=n_splits, shuffle=True, random_state=42), encoded_meta

    print(f"[create_cv_object] Using LeaveOneGroupOut with {n_groups} groups.")
    return LeaveOneGroupOut(), encoded_meta


def perform_optuna_search_grouped_cv(
        base_estimator: BaseEstimator,
        param_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.DataFrame,
        meta: pd.Series,
        n_jobs: int = -1,
        n_trials: int = 512,
        n_splits: int = 5
) -> Tuple[BaseEstimator, Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Performs hyperparameter optimization using Optuna with grouped cross-validation.

    Args:
        base_estimator (BaseEstimator): The base scikit-learn estimator.
        param_dict (Dict[str, Any]): Hyperparameter grid for Optuna.
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target matrix.
        meta (pd.Series): Metadata for grouping.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        n_trials (int, optional): Number of Optuna trials. Defaults to 256.
        n_splits (int, optional): Number of splits for KFold. Defaults to 5.

    Returns:
        Tuple[BaseEstimator, Dict[str, Any], Dict[str, np.ndarray]]: Best model, best parameters, and CV scores.
    """
    print("[perform_optuna_search_grouped_cv] Starting Optuna search...")
    print(f"[perform_optuna_search_grouped_cv] X type: {type(X)}, shape: {X.shape}")
    print(f"[perform_optuna_search_grouped_cv] y type: {type(y)}, shape: {y.shape}")
    print(f"[perform_optuna_search_grouped_cv] meta type: {type(meta)}, shape: {meta.shape}")

    cv, encoded_meta = create_cv_object(meta, n_splits)
    print(f"[perform_optuna_search_grouped_cv] Encoded meta type: {type(encoded_meta)}, shape: {encoded_meta.shape}")

    encoded_meta = encoded_meta.ravel()



    smape_scorer = make_scorer(
        symmetric_mean_absolute_percentage_error,
        greater_is_better=False
    )
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    model_sampler = None
    if isinstance(base_estimator, (XGBRegressor, RandomForestRegressor, MultiOutputRegressor)):
        model_sampler = GPSampler(n_startup_trials=32)
    elif isinstance(base_estimator, LinearRegression) or isinstance(base_estimator, MultiTaskLasso):
        model_sampler = None
    else:
        model_sampler = TPESampler(n_startup_trials=32)

    param_distributions = create_optuna_param_distributions(param_dict)

    # if isinstance(base_estimator, LinearRegression) or isinstance(base_estimator, MultiTaskLasso):
    #     print("[perform_optuna_search_grouped_cv] Using LinearRegression or MultiTaskLasso with default parameters.")
    #
    #     model = clone(base_estimator)
    #     model.fit(X, y)
    #
    #     scores = cross_val_score(
    #         model, X, y,
    #         cv=cv,
    #         scoring=mse_scorer,
    #         n_jobs=n_jobs,
    #         groups=encoded_meta
    #     )
    #     cv_scores = {'per_fold_mse': -scores}
    #     print(f"[perform_optuna_search_grouped_cv] CV scores: {cv_scores}")
    #     return model, model.get_params(), cv_scores

    def objective(trial):
        params = {}
        for name, dist in param_distributions.items():
            if isinstance(dist, optuna.distributions.CategoricalDistribution):
                params[name] = trial.suggest_categorical(name, dist.choices)
            elif isinstance(dist, optuna.distributions.IntDistribution):
                params[name] = trial.suggest_int(name, dist.low, dist.high)
            elif isinstance(dist, optuna.distributions.FloatDistribution):
                params[name] = trial.suggest_float(name, dist.low, dist.high, log=dist.log)
            else:
                raise ValueError(f"Unsupported distribution type for parameter {name}")

        model_instance = clone(base_estimator)
        model_instance.set_params(**params)
        if not supports_multi_output(base_estimator) and y.shape[1] > 1:
            print(
                "[perform_optuna_search_grouped_cv] Estimator does NOT support multi-output. Wrapping with MultiOutputRegressor.")
            model_instance = MultiOutputRegressor(model_instance)
        scores = cross_val_score(
            model_instance, X, y,
            cv=cv, scoring=smape_scorer,
            n_jobs=n_jobs, groups=encoded_meta
        )
        mean_score = -np.mean(scores)  # We minimize SMAPE, so invert sign
        print(f"[Optuna Objective] Trial {trial.number}: Mean SMAPE={mean_score}")
        return mean_score

    study = optuna.create_study(direction='minimize', sampler=model_sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=8)

    importances = optuna.importance.get_param_importances(study)
    print(f"[perform_optuna_search_grouped_cv] Parameter importances:\n{importances}")

    best_params = study.best_params
    print(f"[perform_optuna_search_grouped_cv] Best Hyperparameters: {best_params}")

    best_model = clone(base_estimator)
    best_model.set_params(**best_params)

    best_scores = cross_val_score(
        best_model, X, y,
        cv=cv, scoring=mse_scorer,
        n_jobs=n_jobs, groups=encoded_meta
    )

    cv_scores = {
        'per_fold_mse': [-score for score in best_scores]
    }
    print(f"[perform_optuna_search_grouped_cv] Best model CV scores: {cv_scores}")

    best_model.fit(X, y)

    return best_model, best_params, cv_scores



def perform_optuna_search_LOOcv(
        model_class,
        param_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        meta: Any,
        n_jobs: int = -1,
        n_trials: int = 50
) -> Tuple[BaseEstimator, Dict[str, Any], Dict[str, Any]]:
    param_distributions = create_optuna_param_distributions(param_dict)

    def objective(trial):
        params = {name: trial._suggest(name, dist) for name, dist in param_distributions.items()}
        model_instance = model_class(**params)
        scores = cross_val_score(model_instance, X, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error',
                                 n_jobs=n_jobs)
        return -np.mean(scores)  # We minimize the negative MSE

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = model_class(**best_params)
    best_model.fit(X, y)

    # Calculate per-fold MSE scores for the best model
    loo = LeaveOneOut()
    per_fold_mse = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        best_model.fit(X_train, y_train)
        pred = best_model.predict(X_test)
        mse = ((pred - y_test) ** 2).mean()
        per_fold_mse.append(mse)

    cv_scores = {
        'per_fold_mse': per_fold_mse
    }

    return best_model, best_params, cv_scores

def perform_grid_search_cv(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    meta: Any,
    n_jobs: int = -1
) -> Tuple[BaseEstimator, Dict[str, Any], Dict[str, Any]]:
    """
    Perform Leave-One-Out cross-validation using GridSearchCV with MSE as the scoring metric.

    Parameters:
    - model: The machine learning estimator (e.g., RandomForestRegressor).
    - param_grid: Dictionary with parameter names (`str`) as keys and lists of parameter settings to try as values.
    - X: Feature dataset as a pandas DataFrame.
    - y: Target variable as a pandas Series.
    - n_jobs: Number of jobs to run in parallel. -1 means using all available processors.

    Returns:
    - best_model: Estimator with the best found parameters.
    - best_params: Parameter setting that gave the best results.
    - cv_scores: Dictionary containing per-fold MSE scores for the best parameter set.
    """
    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    def count_param_combinations(param_grid):
        param_lists = [v for v in param_grid.values()]
        return len(list(product(*param_lists)))

    combination_threshold = 500

    n_combinations = count_param_combinations(param_grid)

    if n_combinations <= combination_threshold:
        # Use GridSearchCV
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=loo,
            scoring='neg_mean_squared_error',  # Use built-in negative MSE scorer
            n_jobs=n_jobs,
            return_train_score=False,
            refit=True
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            cv=loo,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            return_train_score=False,
            refit=True,
            n_iter=500,
            random_state=0
        )
    search.fit(X,y)
    best_model = search.best_estimator_
    best_params = search.best_params_

    best_index = search.best_index_

    per_fold_mse = []

    for split_idx in range(loo.get_n_splits(X)):
        split_key = f'split{split_idx}_test_score'
        if split_key in search.cv_results_:
            neg_mse = search.cv_results_[split_key][best_index]
            mse = -neg_mse
            per_fold_mse.append(mse)
        else:
            per_fold_mse.append(np.nan)

    cv_scores = {
        'per_fold_mse': per_fold_mse
    }

    return best_model, best_params, cv_scores