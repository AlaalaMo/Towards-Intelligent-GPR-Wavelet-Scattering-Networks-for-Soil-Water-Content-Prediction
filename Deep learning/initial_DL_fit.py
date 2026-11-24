from akela_shared.csv_loader import get_csv_files, load_csv_files
from akela_shared.splitting import stratified_train_test_split_80_20
import pandas as pd
import torch
from argparse import Namespace
import argparse
import os
from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from train_loop import simple_train_model, evaluate_model_with_dataset
import json
import optuna
from model.base_cnn import Base_CNN, Stand_CNN
from model.base_tf import Stand_Transformer
from model.base_rnn import Stand_RNN
import statistics
import math
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
def split_real_imaginary(X):
    real = np.real(X)
    imaginary = np.imag(X)
    return np.concatenate([real, imaginary], axis=1)
def create_directory_structure(base_path):
    os.makedirs(os.path.join(base_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val'), exist_ok=True)

# Save datasets using pickle
def save_dataset(dataset, path):
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
def load_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_datasets(args):
    base_path = './split'
    train_path = os.path.join(base_path, 'train', 'dataset.pkl')
    test_path = os.path.join(base_path, 'test', 'dataset.pkl')
    val_path = os.path.join(base_path, 'val', 'dataset.pkl')

    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(val_path):
        print("Loading saved datasets...")
        train_dataset = load_dataset(train_path)
        test_dataset = load_dataset(test_path)
        val_dataset = load_dataset(val_path)
    else:
        print("Datasets not found. Creating new splits...")
        csv_files = get_csv_files(data_folder_base='../Final_data', domain=args.domain)
        dataframes = load_csv_files(csv_files, args)
        df = pd.concat(dataframes, axis=0)
        y_df = df['y'].iloc[:,2:-2]

        X = df.loc[:,
            (df.columns.get_level_values(0) != 'y') &
            (df.columns.get_level_values(0) != 'meta') &
            (df.columns.get_level_values(0) != 'height_1') &
            (df.columns.get_level_values(0) != 'height_3')
            ]
        scaler = MinMaxScaler()
        X_real_normalized = pd.DataFrame(scaler.fit_transform(np.real(X)), columns=X.columns)
        X_imag_normalized = pd.DataFrame(scaler.fit_transform(np.imag(X)), columns=X.columns)

        # Concatenate the normalized real and imaginary parts
        X_normalized = pd.concat([X_real_normalized, X_imag_normalized], axis=1)

        # Now X is normalized for both real and imaginary components
        X = X_normalized
        meta_column = df['meta']

        X_train, Y_train, meta_train, X_test, Y_test, meta_test = stratified_train_test_split_80_20(X, y_df,
                                                                                                    meta_column,
                                                                                                    random_seed=args.random_seed)
        X_train, Y_train, meta_train, X_val, Y_val, meta_val = stratified_train_test_split_80_20(X_train, Y_train,
                                                                                                 meta_train,
                                                                                                 random_seed=args.random_seed)
        train_dataset = CustomDataset(X_train, Y_train)
        test_dataset = CustomDataset(X_test, Y_test)
        val_dataset = CustomDataset(X_val, Y_val)

        create_directory_structure(base_path)
        save_dataset(train_dataset, train_path)
        save_dataset(test_dataset, test_path)
        save_dataset(val_dataset, val_path)
    return train_dataset, val_dataset, test_dataset
    # manamage paramaters
# NEW
def evaluate_and_print(model, dataset, dataset_name: str, batch_size: int = 32):
    """Wrapper to evaluate a model on a dataset and print all metrics.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): Dataset for evaluation.
        dataset_name (str): Name identifier (e.g., 'validation', 'test').
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: Dictionary containing all metrics (overall, elementwise, SE, R²).
    """
    (
        rmse,
        mape,
        per_elem_rmse,
        per_elem_mape,
        r2,
        overall_se,
        elem_se,
    ) = evaluate_model_with_dataset(model, dataset, batch_size)

    print(f"\n=== Evaluation on {dataset_name} Dataset ===")
    print(
        f"Overall RMSE: {rmse:.6f} (SE: {overall_se['rmse_se']:.6f})\n"
        f"Overall MAPE: {mape:.6f} (SE: {overall_se['mape_se']:.6f})\n"
        f"R²:           {r2:.6f}"
    )

    k = min(8, len(per_elem_rmse))
    if k > 0:
        print(f"\nElementwise (first {k}) metrics:")
        for i in range(k):
            e = f"element_{i}"
            print(
                f"  {e}: RMSE={per_elem_rmse[e]:.6f} "
                f"(SE={elem_se[e]['rmse_se']:.6f}), "
                f"MAPE={per_elem_mape[e]:.6f} "
                f"(SE={elem_se[e]['mape_se']:.6f})"
            )

    return {
        "dataset": dataset_name,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "overall_se": overall_se,
        "per_element_rmse": per_elem_rmse,
        "per_element_mape": per_elem_mape,
        "per_element_se": elem_se,
    }

def objective(trial):
    global best_model, best_score, best_per_element
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", search_space["learning_rate"]["min"], search_space["learning_rate"]["max"])
    batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"]["values"])
    num_epochs = trial.suggest_categorical("num_epochs", search_space["num_epochs"]["values"])
    print(
        f"below Hyperparameters:\n"
        f"  Learning Rate: {learning_rate}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Num Epochs: {num_epochs}", flush=True
    )
    # Create and train model
    model = nn_model(print_shape=False)
    # model.load_state_dict(torch.load("task_1_best_model.pth"))

    model , _, _ = train_function(learning_rate, model, train_dataset, num_epochs, batch_size)

    # Evaluate model
    # Within your training/validation loop, after each epoch

    # Evaluate on validation dataset
    val_results = evaluate_and_print(
        model=model,
        dataset=val_dataset,
        dataset_name="Validation",
        batch_size=batch_size,
    )

    rmse = val_results["rmse"]
    mape = val_results["mape"]
    per_element_rmse = val_results["per_element_rmse"]
    per_element_mape = val_results["per_element_mape"]

    # Best model tracking
    if rmse < best_score:
        best_score = rmse
        best_model = model
        best_per_element = per_element_rmse

        print("overall rmse:", rmse)
        print("overall mape:", mape)
        print("per element rmse:", per_element_rmse)
        print("per element mape:", per_element_mape)
        print("BEST^")

        # Evaluate best model on train dataset
        train_results = evaluate_and_print(
            model=model,
            dataset=train_dataset,
            dataset_name="Train",
            batch_size=batch_size,
        )

        # Evaluate best model on test dataset
        test_results = evaluate_and_print(
            model=model,
            dataset=test_dataset,
            dataset_name="Test",
            batch_size=batch_size,
        )

    print(
        f"above Hyperparameters:\n"
        f"  Learning Rate: {learning_rate}\n"
        f"  Batch Size: {batch_size}\n"
        f"  Num Epochs: {num_epochs}",
        flush=True,
    )

    return rmse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization with dynamic training function")
    parser.add_argument("--train_function", type=str, required=True, help="Name of the training function to use")
    parser.add_argument("--nn_model", type=str, required=True, help="Name of the nn_model function to use")
    parser.add_argument("--search_space", type=str, required=True, help="Path to JSON file defining the search space")
    parser.add_argument("--domain", type=str, required=True, help="Path to dataset if not yet creayed")
    parser.add_argument("--random_seed", type=int, required=True, help="random_seed")

    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = load_datasets(args)
    with open(args.search_space, "r") as f:
        search_space = json.load(f)
    if args.train_function not in globals():
        raise ValueError(f"Training function '{args.train_function}' not found.")
    train_function = globals()[args.train_function]
    if args.nn_model not in globals():
        raise ValueError(f"nn_model function '{args.nn_model}' not found.")
    nn_model = globals()[args.nn_model]
    best_model = None
    best_score = float("inf")
    best_per_element = None


    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model =nn_model()
    trainable_params = count_trainable_parameters(model)
    print(f"Number of trainable parameters: {trainable_params}")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

    print("Best parameters:", study.best_params)
    print("Best RMSE:", best_score)
    print("Best RMSE per element:", best_per_element)
    torch.save(best_model.state_dict(), "task_10_best_model.pth")
    # train_rmse_list = []
    # train_mape_list = []
    # test_rmse_list = []
    # test_mape_list = []
    # for _ in range(5):
    #     model = nn_model()
    #     model, train_rmse, train_mape = train_function(0.0005950325754821655, model, train_dataset, 10000, 8)
    #     # train_rmse, _ =evaluate_model_with_dataset(model, train_dataset, 4)
    #     train_rmse_list.append(train_rmse)
    #     train_mape_list.append(train_mape)
    #     # val_rmse, per_element_rmse = evaluate_model_with_dataset(model, val_dataset, 4)
    #     test_rmse, test_mape, _ = evaluate_model_with_dataset(model, test_dataset, 4)
    #     test_rmse_list.append(test_rmse)
    #     test_mape_list.append(test_mape)
    # # Compute mean, standard deviation, and standard error for train and test RMSE
    # train_rmse_mean = statistics.mean(train_rmse_list)
    # train_rmse_std = statistics.stdev(train_rmse_list)
    # train_rmse_se = train_rmse_std / math.sqrt(len(train_rmse_list))
    #
    # test_rmse_mean = statistics.mean(test_rmse_list)
    # test_rmse_std = statistics.stdev(test_rmse_list)
    # test_rmse_se = test_rmse_std / math.sqrt(len(test_rmse_list))
    #
    # # Compute mean, standard deviation, and standard error for train and test MAPE
    # train_mape_mean = statistics.mean(train_mape_list)
    # train_mape_std = statistics.stdev(train_mape_list)
    # train_mape_se = train_mape_std / math.sqrt(len(train_mape_list))
    #
    # test_mape_mean = statistics.mean(test_mape_list)
    # test_mape_std = statistics.stdev(test_mape_list)
    # test_mape_se = test_mape_std / math.sqrt(len(test_mape_list))
    #
    # # Print the results
    # print(f"Train RMSE - Mean: {train_rmse_mean:.4f}, Std: {train_rmse_std:.4f}, SE: {train_rmse_se:.4f}")
    # print(f"Train MAPE - Mean: {train_mape_mean:.4f}, Std: {train_mape_std:.4f}, SE: {train_mape_se:.4f}")
    # print(f"Test RMSE - Mean: {test_rmse_mean:.4f}, Std: {test_rmse_std:.4f}, SE: {test_rmse_se:.4f}")
    # print(f"Test MAPE - Mean: {test_mape_mean:.4f}, Std: {test_mape_std:.4f}, SE: {test_mape_se:.4f}")