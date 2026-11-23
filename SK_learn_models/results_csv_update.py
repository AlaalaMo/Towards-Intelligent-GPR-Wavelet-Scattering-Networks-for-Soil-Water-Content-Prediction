import os
import csv
import json
import shutil
import tempfile
import logging
from datetime import datetime
import numpy as np
from typing import Dict, Any, List
import portalocker
from portalocker.exceptions import LockException  # Exception raised when lock cannot be acquired
import random
import time
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("update_results.log"),
        logging.StreamHandler()
    ]
)


def make_json_serializable(obj):
    """Convert supported NumPy values for JSON serialization.

    Args:
        obj: NumPy array, numeric scalar, container, or other value.

    Returns:
        Converted value. Recurses through dictionary values, lists, and
        tuples and passes other values through.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj


def define_fieldnames(y_columns: List[str]) -> List[str]:
    """
    Define the fieldnames for the CSV based on the structure of the data and metrics.

    Args:
        y_columns (List[str]): List of target column names.

    Returns:
        List[str]: List of fieldnames for the CSV.
    """
    base_fields = [
        'config', 'model_type', 'model_params', 'best_model_params',
        'pca_applied', 'r_pca_applied', 'height_column_type', 'meta_location_type',
        'date_time', 'cv_scores','test_adjusted_r2'
    ]

    metric_fields = []
    for set_type in ['train', 'test']:
        for metric in ['rmse', 'mape']:
            metric_fields.extend([
                f'{set_type}_{metric}_overall',
                f'{set_type}_{metric}_overall_se'
            ])

    bootstrap_fields = [
        'bootstrap_metrics_file',
        'bootstrap_rmse_overall_mean',
        'bootstrap_rmse_overall_std',
        'bootstrap_mape_overall_mean',
        'bootstrap_mape_overall_std'
    ]

    per_element_fields = []
    for set_type in ['train', 'test']:
        for y_col in y_columns:
            per_element_fields.extend([
                f'{set_type}_rmse_{y_col}',
                f'{set_type}_mape_{y_col}',
                f'{set_type}_rmse_se_{y_col}',
                f'{set_type}_mape_se_{y_col}',
                f'{set_type}_value_{y_col}',
                f'{set_type}_metric_type_{y_col}',
                f'{set_type}_type_{y_col}'
            ])

    bootstrap_per_element_fields = []
    for y_col in y_columns:
        for metric in ['rmse', 'mape']:
            bootstrap_per_element_fields.extend([
                f'bootstrap_{metric}_{y_col}_mean',
                f'bootstrap_{metric}_{y_col}_std'
            ])

    return base_fields + metric_fields + bootstrap_fields + per_element_fields + bootstrap_per_element_fields


def update_results_csv(
        csv_path: str,
        fieldnames: List[str],
        config: Dict[str, Any],
        model_params: Dict[str, Any],
        metrics: Dict[str, Any],
        pca_applied: bool,
        r_pca_applied: bool,
        height_column_type: str,
        meta_location_type: str,
):
    """Write a fitting result to the shared CSV while holding a file lock.

    Args:
        csv_path: Destination path, including a parent directory.
        fieldnames: Output columns in write order.
        config: Selected preprocessing configuration.
        model_params: Model name, search settings, and selected parameters.
        metrics: Train and test metrics, bootstrap path, and CV scores.
        pca_applied: Whether PCA was applied.
        r_pca_applied: Value to record for the reverse PCA flag.
        height_column_type: Height subset label.
        meta_location_type: Location subset label.

    Raises:
        LockException: The final lock acquisition attempt fails.
        OSError: A directory or file operation fails.

    Notes:
        Backs up the destination and rewrites it via a temporary file.
        Replaces a matching row when its test RMSE improves. Appends a
        row when no existing row is replaced.
    """
    def prepare_row() -> Dict[str, Any]:
        """Build one CSV row with serialized configuration and metrics."""
        row = {
            'config': json.dumps(make_json_serializable(config), sort_keys=True),
            'model_type': model_params['model_type'],
            'model_params': json.dumps(make_json_serializable(model_params['params']), sort_keys=True),
            'best_model_params': json.dumps(make_json_serializable(model_params['best_params']), sort_keys=True),
            'pca_applied': pca_applied,
            'r_pca_applied': r_pca_applied,
            'height_column_type': height_column_type,
            'meta_location_type': meta_location_type,
            'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'cv_scores': json.dumps(make_json_serializable(metrics['cv_scores']), sort_keys=True),
        }
        row[f'test_adjusted_r2'] = metrics['test']['adjusted_r2']
        for set_type in ['train', 'test']:
            row[f'{set_type}_rmse_overall'] = metrics[set_type]['rmse_overall']
            row[f'{set_type}_mape_overall'] = metrics[set_type]['mape_overall']
            row[f'{set_type}_rmse_overall_se'] = metrics[set_type]['rmse_overall_se']
            row[f'{set_type}_mape_overall_se'] = metrics[set_type]['mape_overall_se']

            for y_col in metrics[set_type]['y_columns']:
                rmse, mape, rmse_se, mape_se, _ = metrics[set_type]['per_element'][y_col]
                row[f'{set_type}_rmse_{y_col}'] = rmse
                row[f'{set_type}_mape_{y_col}'] = mape
                row[f'{set_type}_rmse_se_{y_col}'] = rmse_se
                row[f'{set_type}_mape_se_{y_col}'] = mape_se

        row['bootstrap_metrics_file'] = metrics['test'].get('bootstrap_metrics_file', '')

        return row

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_row = prepare_row()

    for attempt in range(1, 100 + 1):
        try:
            # Attempt to acquire the file lock with a timeout of 10 seconds
            with portalocker.Lock(csv_path, timeout=10, mode='a+', flags=portalocker.LOCK_EX):
                # Move the file pointer to the beginning for reading
                file_exists = os.path.exists(csv_path)
                file_is_empty = file_exists and os.path.getsize(csv_path) == 0

                if file_exists:
                    # Create a backup of the existing CSV
                    backup_path = f"{csv_path}.backup"
                    shutil.copy(csv_path, backup_path)
                    print(f"Backup created at {backup_path}")

                # Create a temporary file to write updates
                with tempfile.NamedTemporaryFile('w+', delete=False, newline='', encoding='utf-8') as tmpfile:
                    writer = csv.DictWriter(tmpfile, fieldnames=fieldnames)
                    writer.writeheader()

                    new_row = prepare_row()

                    if not file_exists or file_is_empty:
                        # If the file doesn't exist or is empty, write the new row directly
                        writer.writerow({field: new_row.get(field, 'N/A') for field in fieldnames})
                        print(f"Created new CSV with headers at {csv_path}")
                    else:
                        # Read existing rows from the CSV
                        with open(csv_path, 'r', newline='', encoding='utf-8') as existing_file:
                            reader = csv.DictReader(existing_file)
                            rows = list(reader)

                        row_updated = False
                        for row in rows:
                            # Check if the current row matches the criteria for updating
                            if (row['config'] == new_row['config'] and
                                    row['model_type'] == new_row['model_type'] and
                                    row['model_params'] == new_row['model_params'] and
                                    row['pca_applied'] == str(new_row['pca_applied']) and
                                    row['r_pca_applied'] == str(new_row['r_pca_applied']) and
                                    row['height_column_type'] == new_row['height_column_type'] and
                                    row['meta_location_type'] == new_row['meta_location_type']):

                                # Compare RMSE values to decide whether to update the row
                                try:
                                    existing_rmse = float(row['test_rmse_overall'])
                                    new_rmse = float(new_row['test_rmse_overall'])
                                except ValueError:
                                    print("Invalid RMSE value. Skipping row update.")
                                    writer.writerow(row)
                                    continue

                                if new_rmse < existing_rmse:
                                    writer.writerow(new_row)
                                    row_updated = True
                                    print(f"Existing row updated with lower test RMSE in {csv_path}")
                                else:
                                    writer.writerow(row)
                                    print("Existing row not updated since new test RMSE is not lower.")
                            else:
                                writer.writerow(row)

                        if not row_updated:
                            # Append the new row if no existing row was updated
                            writer.writerow(new_row)
                            print(f"New row appended to {csv_path}")

                # Replace the original CSV with the updated temporary file
                shutil.move(tmpfile.name, csv_path)

            # If the lock was acquired and the update was successful, exit the function
            print(f"Results successfully updated in {csv_path}")
            return  # Success

        except LockException:
            if attempt == 100:
                print(f"Failed to acquire lock after {100} attempts.")
                raise  # Re-raise the exception after the final attempt

            # Calculate exponential backoff delay with some jitter
            delay = .1 * (2 ** (attempt)) + random.uniform(0, 1)
            print(f"Lock already held. Attempt {attempt}/{100}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)