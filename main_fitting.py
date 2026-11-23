import argparse
from deltar2 import run_delta_r2_permutation_multi, run_rf_importance
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import hashlib
import pickle
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
import importlib
from joblib import Parallel, delayed
from loading_in_config import get_transformed_data, get_subset_columns  # Adjust import paths accordingly
from SK_learn_models.test_train_split import test_train_80_20, stratified_train_test_split_80_20
from SK_learn_models.results_csv_update import update_results_csv,define_fieldnames
from SK_learn_models.SK_bayes_CV import perform_optuna_search_grouped_cv as perform_grid_search_cv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error
def calculate_metrics(y_true, y_pred, num_features):
    """
    Calculate RMSE, MAPE, and Adjusted R^2 along with their standard errors.

    :param y_true: True values
    :param y_pred: Predicted values
    :param num_features: Number of features in the model
    :return: RMSE, MAPE, RMSE standard error, MAPE standard error, Adjusted R^2
    """
    errors = y_true - y_pred
    squared_errors = errors ** 2
    absolute_percentage_errors = np.abs(errors / y_true) * 100

    # Calculate RMSE and MAPE
    rmse = np.sqrt(np.mean(squared_errors))
    mape = np.mean(absolute_percentage_errors)

    # Calculate standard errors
    rmse_se = np.std(np.sqrt(squared_errors.flatten())) / np.sqrt(y_true.size)
    mape_se = np.std(absolute_percentage_errors.flatten()) / np.sqrt(y_true.size)

    # Calculate R^2
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum(squared_errors)
    r2 = 1 - (ss_residual / ss_total)

    # Calculate Adjusted R^2
    n = y_true.size
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))

    return rmse, mape, rmse_se, mape_se, str(num_features)+', '+ str(r2)+", "+", "+str(mean_absolute_error(y_true=y_true,y_pred=y_pred))

def single_bootstrap_iteration(model, X_test, Y_test, y_columns, n_samples):
    """
    Perform a single bootstrap iteration.

    :param model: Trained model
    :param X_test: Test features (NumPy array)
    :param Y_test: Test targets (NumPy array)
    :param y_columns: List of target column names
    :param n_samples: Number of samples in the test set
    :return: overall_rmse, overall_mape, per_element_metrics
    """
    # Ensure inputs are NumPy arrays
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    # Sample indices with replacement
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_bootstrap = X_test[indices]
    Y_bootstrap = Y_test[indices]

    # Make predictions
    Y_pred_bootstrap = model.predict(X_bootstrap)

    # Calculate overall metrics
    overall_rmse, overall_mape, _, _, _ = calculate_metrics(
        Y_bootstrap, Y_pred_bootstrap,X_test.shape[1]
    )

    # Calculate per-element metrics
    per_element_metrics = {}
    for idx, y_col in enumerate(y_columns):
        rmse, mape, _, _, _ = calculate_metrics(
            Y_bootstrap[:, idx], Y_pred_bootstrap[:, idx], X_test.shape[1]
        )
        per_element_metrics[y_col] = {'rmse': rmse, 'mape': mape}

    return overall_rmse, overall_mape, per_element_metrics

def bootstrap_test_set(model, X_test, Y_test, n_bootstraps=1000, y_columns=None, n_jobs=-1):
    """
    Perform bootstrapping on the test set and calculate metrics per bootstrap sample in parallel.

    :param model: Trained model
    :param X_test: Test features
    :param Y_test: Test targets
    :param n_bootstraps: Number of bootstrap iterations
    :param y_columns: List of target column names
    :param n_jobs: Number of jobs for parallel processing (-1 uses all cores)
    :return: Dictionary containing per-bootstrap metrics
    """
    n_samples = X_test.shape[0]

    # Use Parallel to perform bootstrapping in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap_iteration)(model, X_test, Y_test, y_columns, n_samples)
        for _ in range(n_bootstraps)
    )

    # Initialize the metrics dictionary
    bootstrap_metrics = {
        'rmse_overall': [],
        'mape_overall': [],
        'per_element': {y_col: {'rmse': [], 'mape': []} for y_col in y_columns}
    }

    # Collect the results
    for overall_rmse, overall_mape, per_element_metrics in results:
        bootstrap_metrics['rmse_overall'].append(overall_rmse)
        bootstrap_metrics['mape_overall'].append(overall_mape)
        for y_col in y_columns:
            bootstrap_metrics['per_element'][y_col]['rmse'].append(per_element_metrics[y_col]['rmse'])
            bootstrap_metrics['per_element'][y_col]['mape'].append(per_element_metrics[y_col]['mape'])

    return bootstrap_metrics

def generate_unique_filename(config, height_column_type, meta_location_type, extension='pkl'):
    """
    Generate a unique filename based on configuration and subset columns.

    :param config: Configuration dictionary
    :param height_column_type: Type of the height column
    :param meta_location_type: Type of the meta location
    :param extension: File extension (default: 'pkl')
    :return: Unique filename as a string
    """
    # Convert config to a sorted JSON string to ensure consistency
    config_str = json.dumps(config, sort_keys=True)

    # Create a unique hash of the config string
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()

    # Construct the filename with a timestamp to ensure uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"bootstrap_{height_column_type}_{meta_location_type}_{config_hash}_{timestamp}.{extension}"

    return filename

# def update_results_csv(
#         csv_path: str,
#         config: Dict[str, Any],
#         model_params: Dict[str, Any],
#         metrics: Dict[str, Any],
#         pca_applied: bool,
#         r_pca_applied: bool,
#         height_column_type: str,
#         meta_location_type: str
# ):
#     """
#     Update the results CSV with the given metrics and additional information.
#
#     If the CSV or its directory does not exist, they will be created.
#     If a row with the same configuration exists, it will only be updated if the new
#     test_rmse_overall is lower than the existing one.
#
#     Args:
#         csv_path (str): Path to the CSV file.
#         config (Dict[str, Any]): Configuration dictionary.
#         model_params (Dict[str, Any]): Model parameters dictionary.
#         metrics (Dict[str, Any]): Metrics dictionary containing overall and per-element metrics.
#         pca_applied (bool): Whether PCA was applied.
#         r_pca_applied (bool): Whether reverse PCA was applied.
#         height_column_type (str): Type of the height column.
#         meta_location_type (str): Type of the meta location.
#     """
#     # Convert configurations to JSON strings for consistent comparison
#     FIXED_FIELDNAMES = [
#         'config',
#         'model_type',
#         'model_params',
#         'best_model_params',
#         'pca_applied',
#         'r_pca_applied',
#         'height_column_type',
#         'meta_location_type',
#         'train_rmse_overall',
#         'train_rmse_overall_se',
#         'train_mape_overall',
#         'train_mape_overall_se',
#         'test_rmse_overall',
#         'test_rmse_overall_se',
#         'test_mape_overall',
#         'test_mape_overall_se',
#         'cv_scores',
#         'date_time',
#         'bootstrap_metrics_file',
#         # Add any additional fixed columns here
#         # e.g., 'bootstrap_rmse_overall_mean', 'bootstrap_rmse_overall_std', etc.
#     ]
#     config_str = json.dumps(config, sort_keys=True)
#     model_params_str = json.dumps(model_params, sort_keys=True)
#     best_model_params_str = json.dumps(model_params['best_params'], sort_keys=True)
#
#     # Get current date and time
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#     # Prepare the new row with cv_scores included
#     new_row = {
#         'config': config_str,
#         'model_type': model_params['model_type'],
#         'model_params': model_params_str,
#         'best_model_params': best_model_params_str,
#         'pca_applied': pca_applied,
#         'r_pca_applied': r_pca_applied,
#         'height_column_type': height_column_type,
#         'meta_location_type': meta_location_type,
#         'train_rmse_overall': metrics['train']['rmse_overall'],
#         'train_rmse_overall_se': metrics['train'].get('rmse_overall_se', ''),
#         'train_mape_overall': metrics['train']['mape_overall'],
#         'train_mape_overall_se': metrics['train'].get('mape_overall_se', ''),
#         'test_rmse_overall': metrics['test']['rmse_overall'],
#         'test_rmse_overall_se': metrics['test'].get('rmse_overall_se', ''),
#         'test_mape_overall': metrics['test']['mape_overall'],
#         'test_mape_overall_se': metrics['test'].get('mape_overall_se', ''),
#         'cv_scores': json.dumps(metrics.get('cv_scores', {})),  # Store cv_scores as JSON
#         'date_time': current_datetime,
#         'bootstrap_metrics_file': metrics['test'].get('bootstrap_metrics_file', '')  # Include the file path
#     }
#
#     # Add bootstrap summary statistics to the new row
#     if 'bootstrap' in metrics['test']:
#         bootstrap_rmse_overall_mean = np.mean(metrics['test']['bootstrap']['rmse_overall'])
#         bootstrap_rmse_overall_std = np.std(metrics['test']['bootstrap']['rmse_overall'])
#         bootstrap_mape_overall_mean = np.mean(metrics['test']['bootstrap']['mape_overall'])
#         bootstrap_mape_overall_std = np.std(metrics['test']['bootstrap']['mape_overall'])
#
#         new_row.update({
#             'bootstrap_rmse_overall_mean': bootstrap_rmse_overall_mean,
#             'bootstrap_rmse_overall_std': bootstrap_rmse_overall_std,
#             'bootstrap_mape_overall_mean': bootstrap_mape_overall_mean,
#             'bootstrap_mape_overall_std': bootstrap_mape_overall_std
#         })
#
#         # Add per-element bootstrap metrics
#         y_columns = metrics['test']['y_columns']
#         for y_col in y_columns:
#             rmse_list = metrics['test']['bootstrap']['per_element'][y_col]['rmse']
#             mape_list = metrics['test']['bootstrap']['per_element'][y_col]['mape']
#             new_row[f'bootstrap_rmse_{y_col}_mean'] = np.mean(rmse_list)
#             new_row[f'bootstrap_rmse_{y_col}_std'] = np.std(rmse_list)
#             new_row[f'bootstrap_mape_{y_col}_mean'] = np.mean(mape_list)
#             new_row[f'bootstrap_mape_{y_col}_std'] = np.std(mape_list)
#
#     # Add per-element metrics for training set
#     for y_col in metrics['train'].get('y_columns', []):
#         per_element = metrics['train']['per_element'].get(y_col, (None, None, None, None))
#         if len(per_element) != 4:
#             raise ValueError(f"Expected 4 metrics for per_element[{y_col}], got {len(per_element)}")
#         rmse, mape, rmse_se, mape_se = per_element
#         new_row[f'train_rmse_{y_col}'] = rmse
#         new_row[f'train_rmse_{y_col}_se'] = rmse_se
#         new_row[f'train_mape_{y_col}'] = mape
#         new_row[f'train_mape_{y_col}_se'] = mape_se
#
#     # Add per-element metrics for test set
#     for y_col in metrics['test'].get('y_columns', []):
#         per_element = metrics['test']['per_element'].get(y_col, (None, None, None, None))
#         if len(per_element) != 4:
#             raise ValueError(f"Expected 4 metrics for per_element[{y_col}], got {len(per_element)}")
#         rmse, mape, rmse_se, mape_se = per_element
#         new_row[f'test_rmse_{y_col}'] = rmse
#         new_row[f'test_rmse_{y_col}_se'] = rmse_se
#         new_row[f'test_mape_{y_col}'] = mape
#         new_row[f'test_mape_{y_col}_se'] = mape_se
#
#     # Ensure the directory exists
#     csv_dir = os.path.dirname(csv_path)
#     if csv_dir and not os.path.exists(csv_dir):
#         os.makedirs(csv_dir, exist_ok=True)
#
#     # Initialize rows
#     rows = []
#     if os.path.exists(csv_path):
#         with open(csv_path, 'r', newline='', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             rows = list(reader)
#
#     row_updated = False
#
#     # Iterate through existing rows to find a matching configuration
#     for row in rows:
#         if (
#                 row.get('config') == config_str and
#                 row.get('model_type') == model_params['model_type'] and
#                 row.get('model_params') == model_params_str and
#                 row.get('pca_applied') == str(pca_applied) and
#                 row.get('r_pca_applied') == str(r_pca_applied) and
#                 row.get('height_column_type') == height_column_type and
#                 row.get('meta_location_type') == meta_location_type
#         ):
#             print('Found matching row with same parameters.')
#             try:
#                 existing_rmse = float(row.get('test_rmse_overall', float('inf')))
#                 new_rmse = float(new_row['test_rmse_overall'])
#             except ValueError:
#                 existing_rmse = float('inf')
#                 new_rmse = float(new_row['test_rmse_overall'])
#
#             # Update the row only if the new test RMSE is lower
#             if new_rmse < existing_rmse:
#                 row.update(new_row)
#                 row_updated = True
#                 print(f"Existing row updated with lower test RMSE in {csv_path}")
#             else:
#                 print(
#                     f"Existing row not updated since new test RMSE ({new_rmse}) is not lower than existing test RMSE ({existing_rmse})")
#                 print(new_row)
#                 return None
#             break  # Assuming unique configuration, exit after finding a match
#
#     if not row_updated:
#         rows.append(new_row)
#         print(f"New row appended to {csv_path}")
#
#     fieldnames = set()
#     for row in rows:
#         fieldnames.update(row.keys())
#     fieldnames = sorted(fieldnames)
#
#     with open(csv_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"Results updated in {csv_path}")

# def parallel_fit_and_score(estimator, X, y, train, test, parameters):
#     """Fit the estimator on the train set and score it on the test set using a weighted sum of normalized MSE and MAPE."""
#     clf = clone(estimator)
#     clf.set_params(**parameters)
#
#     clf.fit(X.iloc[train], y.iloc[train])
#
#     y_pred = clf.predict(X.iloc[test])
#
#     mse = mean_squared_error(y.iloc[test], y_pred, multioutput='raw_values')
#     mape = mean_absolute_percentage_error(y.iloc[test], y_pred, multioutput='raw_values')
#
#     epsilon = 1e-10
#     mse_max = np.max(mse) if np.max(mse) != 0 else epsilon
#     mape_max = np.max(mape) if np.max(mape) != 0 else epsilon
#
#     mse_norm = mse / mse_max
#     mape_norm = mape / mape_max
#
#     weight_mse = 0.6
#     weight_mape = 0.4
#
#     combined_score = (weight_mse * mse_norm) + (weight_mape * mape_norm)
#
#     return combined_score

# def perform_parallel_loo_cv(model: BaseEstimator, param_grid: Dict[str, Any], X: pd.DataFrame, y: pd.DataFrame,
#                             n_jobs: int = -1) -> Tuple[BaseEstimator, Dict[str, Any], Dict[str, Any]]:
#     """Perform Leave-One-Out cross-validation in parallel."""
#     loo = LeaveOneOut()
#     param_list = list(ParameterGrid(param_grid))
#
#     def evaluate_params(params):
#         scores = Parallel(n_jobs=n_jobs)(
#             delayed(parallel_fit_and_score)(
#                 model, X, y, train, test, params
#             ) for train, test in loo.split(X)
#         )
#         scores = np.array(scores)
#         return scores
#
#     # Evaluate all parameter combinations
#     results = Parallel(n_jobs=1)(delayed(evaluate_params)(params) for params in param_list)
#
#     # Calculate mean scores for each parameter set
#     mean_scores = [np.mean(score) for score in results]
#
#     # Select the best parameters
#     best_idx = np.argmin(mean_scores)
#     best_params = param_list[best_idx]
#     best_scores = results[best_idx]  # Per-fold scores for the best parameters
#
#     # Train the best model on the entire training dataset
#     best_model = clone(model).set_params(**best_params)
#     best_model.fit(X, y)
#
#     # Prepare cv_scores dictionary with per-fold scores
#     cv_scores = {
#         'per_fold_mse': best_scores.tolist(),  # Convert to list for JSON serialization
#     }
#
#     return best_model, best_params, cv_scores



def load_model(model_type: str, n_jobs_mor: int = 1) -> BaseEstimator:
    """
    Load a machine learning model by its name. Automatically wraps models that do not
    natively support multi-output regression with MultiOutputRegressor.

    Parameters:
    - model_type (str): The name of the model to load (e.g., 'RandomForestRegressor', 'Lasso', 'XGBRegressor').
    - n_jobs_mor (int): Number of jobs for MultiOutputRegressor parallelism.

    Returns:
    - BaseEstimator: An instance of the requested model, wrapped with MultiOutputRegressor if necessary.

    Raises:
    - ValueError: If the specified model_type is not found in sklearn or xgboost.
    """
    # Define models that require wrapping for multi-output regression
    # Update this set as needed when adding more models that require wrapping
    # models_requiring_wrapping = {'XGBRegressor'}

    def get_multioutput_regressor():
        try:
            mo_module = importlib.import_module('sklearn.multioutput')
            MultiOutputRegressor = getattr(mo_module, 'MultiOutputRegressor')
            return MultiOutputRegressor
        except (ImportError, AttributeError):
            raise ImportError("MultiOutputRegressor could not be imported from sklearn.multioutput.")

    sklearn_modules = ['linear_model', 'ensemble', 'svm', 'tree', 'neighbors']
    for module_name in sklearn_modules:
        try:
            module = importlib.import_module(f'sklearn.{module_name}')
            model_class = getattr(module, model_type)
            model_instance = model_class()

            # if model_type in models_requiring_wrapping:
            #     MultiOutputRegressor = get_multioutput_regressor()
            #     model_instance = MultiOutputRegressor(model_instance, n_jobs=n_jobs_mor)

            return model_instance
        except (ImportError, AttributeError):
            continue

    try:
        xgb_module = importlib.import_module('xgboost')
        model_class = getattr(xgb_module, model_type)
        model_instance = model_class()

        return model_instance
    except (ImportError, AttributeError):
        pass  # Proceed to raise an error

    raise ValueError(f"Model type '{model_type}' is not supported or does not exist in sklearn or xgboost.")

def main():
    parser = argparse.ArgumentParser(description='Process transformed dataset and train model.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file (JSON).')
    parser.add_argument('--config_index', type=int, required=True,
                        help='Index of the desired configuration in the list.')
    parser.add_argument('--apply_pca', action='store_true',
                        help='Apply PCA to preserve 80% variance if not already applied.')
    parser.add_argument('--r_pca', action='store_true', help='Apply R PCA (optional).')
    parser.add_argument('--height_column_type', type=str, required=True,
                        help='Height column type to subset (or "all").')
    parser.add_argument('--location', type=str, required=True,
                        help='Location to subset to from meta column')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Model type from sklearn (e.g., LinearRegression).')
    parser.add_argument('--hyperparams', type=str, required=True,
                        help='Path to JSON file with hyperparameters to search over.')
    parser.add_argument('--results_csv', type=str, default='results.csv', help='Path to the results CSV file.')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of jobs for parallel processing. Default is -1 (all cores).')
    parser.add_argument('--split_function', type=str, required=True,
                        help='Name of the function to split data.')
    parser.add_argument('--n_bootstraps', type=int, default=1000,
                        help='Number of bootstrap iterations for the test set. Default is 1000.')
    parser.add_argument('--bootstrap_folder', type=str, default='bootstrap_metrics',
                        help='Folder to save bootstrap metrics.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42. Set to 0 for full randomness)')

    args = parser.parse_args()
    print(f"Parsed args: {args}")

    # Set random seed if provided
    if args.random_seed is not None:
        print(f"Setting random seed to {args.random_seed}")
        np.random.seed(args.random_seed)
    df, config = get_transformed_data(args.config_file, args.config_index)
    df = df[df[('y', 'avgdepth3')] <= 48]

    config['config_index'] = args.config_index
    df_subset = get_subset_columns(df, args.height_column_type, args.location)
    print(f"DataFrame subsetted. Shape: {df_subset.shape}")

    pca_applied = False
    r_pca_applied = False

    y_columns = [col for col in df_subset.columns if col[0] == 'y']
    if len(y_columns) != 4:
        raise ValueError(f"Expected exactly four 'y' columns. Found: {len(y_columns)}")

    # Subset the 'y' columns
    y_df = df_subset['y']

    # Subset everything besides 'y' columns (this excludes 'meta' as well for now)
    # Subset everything besides 'y' and 'meta' columns
    X = df_subset.loc[:,
        (df_subset.columns.get_level_values(0) != 'y') & (df_subset.columns.get_level_values(0) != 'meta')]
    meta_column = df_subset['meta']

    print(f"Features shape: {X.shape}, Target shape: {y_df.shape}, Meta shape: {meta_column.shape}")
    for col in meta_column.columns:
        unique_values = meta_column[col].unique()
        print(f"Split CV folds on '{col}' unique values: {unique_values}")

    print("Ensure that the meta subset is correctly specified in the loading_in_config file.")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Apply PCA if specified
    if args.apply_pca:
        print("Applying PCA to preserve 90% variance.")
        pca_applied = True
        pca = PCA(n_components=0.80)
        X = pca.fit_transform(X)
        X = pd.DataFrame(X)

    # Check if the split function exists and call it
    func_name = args.split_function
    if func_name in globals():
        split_function = globals()[func_name]
    else:
        raise ValueError(f"Split function '{func_name}' not found in global namespace.")

    # Pass X, y_df, meta_column, and random_seed to the split function for stratified splitting
    X_train, Y_train, meta_train, X_test, Y_test, meta_test = split_function(X, y_df, meta_column, random_seed=args.random_seed)
    print(f"Data split using {func_name}.")
    print(f"Training set shape: X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"Test set shape: X_test={X_test.shape}, Y_test={Y_test.shape}")

    with open(args.hyperparams, 'r') as f:
        hyperparams = json.load(f)
    model = load_model(args.model_type)
    param_grid = hyperparams.get('params', {})

    if not param_grid:
        print("Hyperparameter grid is empty. Please check if valid hyperparameters.")

    print("Using Parallel Leave-One-Out Cross-Validation (LOO CV) on the training dataset.")
    print("Training sets size X_train and Y_train: ",X_train.shape,Y_train.shape)
    best_model, best_params, cv_scores = perform_grid_search_cv(model, param_grid, X_train, Y_train, meta_train, n_jobs=args.n_jobs)

    print(f"Best Hyperparameters: {best_params}")

    # Process cv_scores
    per_fold_rmse = np.sqrt(np.array(cv_scores['per_fold_mse']))
    cv_scores['per_fold_rmse'] = per_fold_rmse.tolist()
    cv_rmse_overall = np.mean(per_fold_rmse)
    print(f"LOO CV RMSE (Overall) on training set: {cv_rmse_overall}")

    y_columns = list(y_df.columns)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_train_np = Y_train.values
    y_test_np = Y_test.values

    y_test_np = Y_test.values  # shape (n_samples, 4) with order [10cm, 20cm, 30cm, 40cm]
    # artifacts = run_delta_r2_permutation_multi(
    #     model=best_model,
    #     X_test=X_test,
    #     y_test=y_test_np,
    #     window_size=128,
    #     stride=32,
    #     f_min_ghz=0.4,
    #     f_max_ghz=2.0,
    #     n_permutations=8,
    #     random_state=42,
    #     out_dir="./results/delta_r2_depthwise",
    #     prefix="wsn_rf_delta_r2_depths",
    #     depth_labels=["10 cm", "20 cm", "30 cm", "40 cm"],
    # )
    # print(json.dumps(artifacts["eda"], indent=2))
    # print(artifacts["paths"])
    # rf_artifacts = run_rf_importance(
    #     model=best_model,          # sklearn RandomForestRegressor
    #     f_min_ghz=0.4,
    #     f_max_ghz=2.0,
    #     out_dir="./results/rf_best",
    #     prefix="wsn_rf_feature_importance",
    #     dpi=200,
    #     use_normalized=False,      # set True to scale 0..1 for visual comparability to ΔR²
    # )
    overall_rmse_train, overall_mape_train, overall_rmse_se_train, overall_mape_se_train, _ = calculate_metrics(
        y_train_np, y_train_pred, X_train.shape[1]
    )
    per_element_metrics_train = {y_col: calculate_metrics(y_train_np[:, idx], y_train_pred[:, idx], X_train.shape[1])
                                 for idx, y_col in enumerate(y_columns)}

    overall_rmse_test, overall_mape_test, overall_rmse_se_test, overall_mape_se_test, test_adjusted_r2 = calculate_metrics(
        y_test_np, y_test_pred, X_test.shape[1])

    per_element_metrics_test = {y_col: calculate_metrics(y_test_np[:, idx], y_test_pred[:, idx],X_train.shape[1])
                                for idx, y_col in enumerate(y_columns)}

    print(f"Final Training Set Metrics:")
    print(f"Overall RMSE: {overall_rmse_train:.4f} ± {overall_rmse_se_train:.4f}")
    print(f"Overall MAPE: {overall_mape_train:.4f}% ± {overall_mape_se_train:.4f}%")
    for y_col, (rmse, mape, rmse_se, mape_se,_) in per_element_metrics_train.items():
        print(f"Metrics for '{y_col}' on training set:")
        print(f"  RMSE: {rmse:.4f} ± {rmse_se:.4f}")
        print(f"  MAPE: {mape:.4f}% ± {mape_se:.4f}%")

    print(f"Final Test Set Metrics:")
    print(f"Overall RMSE: {overall_rmse_test:.4f} ± {overall_rmse_se_test:.4f}")
    print(f"Overall MAPE: {overall_mape_test:.4f}% ± {overall_mape_se_test:.4f}%")
    for y_col, (rmse, mape, rmse_se, mape_se,_) in per_element_metrics_test.items():
        print(f"Metrics for '{y_col}' on test set:")
        print(f"  RMSE: {rmse:.4f} ± {rmse_se:.4f}")
        print(f"  MAPE: {mape:.4f}% ± {mape_se:.4f}%")

    n_bootstraps = args.n_bootstraps
    print(f"Performing bootstrapping on the test set with {n_bootstraps} iterations.")
    bootstrap_metrics = bootstrap_test_set(
        best_model, X_test, Y_test, n_bootstraps=n_bootstraps, y_columns=y_columns, n_jobs=args.n_jobs
    )
    print("Bootstrapping completed.")

    bootstrap_folder = args.bootstrap_folder
    if not os.path.exists(bootstrap_folder):
        os.makedirs(bootstrap_folder)

    bootstrap_filename = generate_unique_filename(config, args.height_column_type, args.location)
    bootstrap_filepath = os.path.join(bootstrap_folder, bootstrap_filename)

    with open(bootstrap_filepath, 'wb') as f:
        pickle.dump(bootstrap_metrics, f)
    print(f"Bootstrap metrics saved to {bootstrap_filepath}")

    metrics = {
        'train': {
            'rmse_overall': overall_rmse_train,
            'rmse_overall_se': overall_rmse_se_train,
            'mape_overall': overall_mape_train,
            'mape_overall_se': overall_mape_se_train,
            'per_element': per_element_metrics_train,
            'y_columns': y_columns
        },
        'test': {
            'rmse_overall': overall_rmse_test,
            'rmse_overall_se': overall_rmse_se_test,
            'mape_overall': overall_mape_test,
            'mape_overall_se': overall_mape_se_test,
            'adjusted_r2' : test_adjusted_r2,
            'per_element': per_element_metrics_test,
            'y_columns': y_columns,
            'bootstrap_metrics_file': bootstrap_filepath
        },
        'cv_scores': cv_scores
    }

    # Update results CSV
    model_params = {
        'model_type': args.model_type,
        'params': param_grid,
        'best_params': best_params
    }
    fieldnames = define_fieldnames(y_columns)
    update_results_csv(args.results_csv, fieldnames, config, model_params, metrics,
                       pca_applied, r_pca_applied, args.height_column_type, args.location)

    # Documentation: How to load the saved bootstrap metrics file
    print("\nTo load the saved bootstrap metrics for further analysis, use the following code:")
    print(f"""
import pickle

bootstrap_metrics_file = '{bootstrap_filepath}'
with open(bootstrap_metrics_file, 'rb') as f:
    bootstrap_metrics = pickle.load(f)
# Now, bootstrap_metrics contains the per-bootstrap metrics
    """)

if __name__ == '__main__':
    main()
