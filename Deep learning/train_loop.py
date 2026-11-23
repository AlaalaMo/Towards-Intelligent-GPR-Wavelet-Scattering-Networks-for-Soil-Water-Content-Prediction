import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from math import sqrt


def simple_train_model(learning_rate, model, dataset, num_epochs, batch_size):
    """
    Trains a PyTorch model on a given dataset and returns the trained model,
    RMSE, and MAPE of the final epoch.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        model (torch.nn.Module): The PyTorch model to be trained.
        dataset (torch.utils.data.Dataset): The dataset for training.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        model (torch.nn.Module): The trained PyTorch model.
        final_rmse (float): RMSE of the final epoch.
        final_mape (float): MAPE of the final epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mape = 0.0
        y_true_list, y_pred_list = [], []

        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_mape += (torch.abs((predictions - targets) / targets)
                           .sum().item())
            y_true_list.append(targets.detach().cpu())
            y_pred_list.append(predictions.detach().cpu())

        # Epoch metrics
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2_score = 1 - ss_res / ss_tot

        epoch_rmse = sqrt(epoch_loss / (len(dataset) * 4))
        epoch_mape /= (len(dataset) * 4)
        print(f"Epoch {epoch + 1}/{num_epochs}, RMSE: {epoch_rmse:.4f}, "
              f"MAPE: {epoch_mape:.4f}, R²: {r2_score:.4f}")

        # Safeguard: terminate if R² too low at epoch 100
        if epoch + 1 >= 100 and r2_score <= 0.15:
            print("Early termination: R² below 0.15 at epoch 100.")
            break

    final_rmse = epoch_rmse
    final_mape = epoch_mape

    return model, final_rmse, final_mape


# CHANGED
def evaluate_model_with_dataset(model, dataset, batch_size=32, eps=1e-8):
    """Evaluate model; print and return RMSE/MAPE (+ SE), elementwise metrics, and R^2.

    Args:
        model (torch.nn.Module): Trained PyTorch model to evaluate.
        dataset (torch.utils.data.Dataset): Dataset for evaluation.
        batch_size (int): Dataloader batch size.
        eps (float): Small constant to avoid division by zero in MAPE.

    Returns:
        float: Overall RMSE.
        float: Overall MAPE.
        dict[str, float]: Elementwise RMSE keyed by 'element_i'.
        dict[str, float]: Elementwise MAPE keyed by 'element_i'.
        float: R^2 over all targets flattened.
        dict[str, float]: Overall standard errors {'rmse_se', 'mape_se'}.
        dict[str, dict[str, float]]: Elementwise SEs keyed by 'element_i' with {'rmse_se','mape_se'}.
    """
    import torch
    from torch.utils.data import DataLoader
    from math import sqrt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    # Overall accumulators (per scalar target z across all samples * elements)
    total_sse = 0.0               # sum(z) where z = (y_hat - y)^2
    total_sse2 = 0.0              # sum(z^2) = sum((err^2)^2) = sum(err^4)
    total_ape = 0.0               # sum(|err| / (|y| + eps))
    total_ape2 = 0.0              # sum(ape^2)
    n_rows = 0                    # number of samples (rows)
    n_scalars = 0                 # total number of scalar targets = rows * elements

    # Elementwise accumulators (vector length = E)
    sse_sum_per_elem = None       # sum over rows of z_j
    sse2_sum_per_elem = None      # sum over rows of z_j^2
    ape_sum_per_elem = None       # sum over rows of ape_j
    ape2_sum_per_elem = None      # sum over rows of ape_j^2

    # R^2 streaming components
    ss_res = 0.0
    sum_y = 0.0
    sum_y2 = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            err = preds - targets
            sse_batch = err.pow(2)                             # (B, ..., E)
            ape_batch = err.abs() / (targets.abs() + eps)      # (B, ..., E)

            # Overall totals
            total_sse += sse_batch.sum().item()
            total_sse2 += sse_batch.pow(2).sum().item()
            total_ape += ape_batch.sum().item()
            total_ape2 += ape_batch.pow(2).sum().item()

            bsz = targets.shape[0]
            E = targets.shape[-1]
            n_rows += bsz
            n_scalars += bsz * E

            # Reduce over all dims except last (elements)
            reduce_dims = tuple(range(0, sse_batch.ndim - 1))
            sse_elem = sse_batch.sum(dim=reduce_dims)          # (E,)
            sse2_elem = sse_batch.pow(2).sum(dim=reduce_dims)  # (E,)
            ape_elem = ape_batch.sum(dim=reduce_dims)          # (E,)
            ape2_elem = ape_batch.pow(2).sum(dim=reduce_dims)  # (E,)

            if sse_sum_per_elem is None:
                sse_sum_per_elem = sse_elem.detach().cpu()
                sse2_sum_per_elem = sse2_elem.detach().cpu()
                ape_sum_per_elem = ape_elem.detach().cpu()
                ape2_sum_per_elem = ape2_elem.detach().cpu()
            else:
                sse_sum_per_elem += sse_elem.detach().cpu()
                sse2_sum_per_elem += sse2_elem.detach().cpu()
                ape_sum_per_elem += ape_elem.detach().cpu()
                ape2_sum_per_elem += ape2_elem.detach().cpu()

            # R^2 accumulators
            y_flat = targets.reshape(-1)
            yhat_flat = preds.reshape(-1)
            ss_res += (y_flat - yhat_flat).pow(2).sum().item()
            sum_y += y_flat.sum().item()
            sum_y2 += y_flat.pow(2).sum().item()

    # Overall means
    mse_overall = total_sse / max(n_scalars, 1)
    rmse_overall = sqrt(mse_overall)
    mape_overall = total_ape / max(n_scalars, 1)

    # Overall SEs via analytic formulae
    # For mean of z: Var(mean) = Var(z)/N with Var(z)=E[z^2]-E[z]^2
    Ez2 = total_sse2 / max(n_scalars, 1)
    var_z = max(Ez2 - mse_overall**2, 0.0)
    se_mse_overall = (var_z / max(n_scalars, 1)) ** 0.5
    rmse_se_overall = se_mse_overall / max(2.0 * rmse_overall, eps)  # delta method

    Eape2 = total_ape2 / max(n_scalars, 1)
    var_ape = max(Eape2 - mape_overall**2, 0.0)
    mape_se_overall = (var_ape / max(n_scalars, 1)) ** 0.5

    # Elementwise means
    import torch as _torch
    mse_elem = sse_sum_per_elem / max(n_rows, 1)               # (E,)
    rmse_elem = _torch.sqrt(_torch.clamp(mse_elem, min=0.0))   # (E,)
    mape_elem = ape_sum_per_elem / max(n_rows, 1)              # (E,)

    # Elementwise SEs
    Ez2_elem = sse2_sum_per_elem / max(n_rows, 1)              # (E,)
    var_z_elem = _torch.clamp(Ez2_elem - mse_elem.pow(2), min=0.0)
    se_mse_elem = _torch.sqrt(var_z_elem / max(n_rows, 1))     # (E,)
    rmse_se_elem = se_mse_elem / _torch.clamp(2.0 * rmse_elem, min=eps)

    Eape2_elem = ape2_sum_per_elem / max(n_rows, 1)
    var_ape_elem = _torch.clamp(Eape2_elem - mape_elem.pow(2), min=0.0)
    mape_se_elem = _torch.sqrt(var_ape_elem / max(n_rows, 1))

    # R^2
    n_total = float(n_scalars)
    mean_y = sum_y / n_total if n_total > 0 else 0.0
    ss_tot = max(sum_y2 - n_total * (mean_y ** 2), 0.0)
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # Dicts
    elementwise_rmse_dict = {f"element_{i}": v for i, v in enumerate(rmse_elem.tolist())}
    elementwise_mape_dict = {f"element_{i}": v for i, v in enumerate(mape_elem.tolist())}
    elementwise_se_dict = {
        f"element_{i}": {"rmse_se": rmse_se_elem[i].item(), "mape_se": mape_se_elem[i].item()}
        for i in range(len(rmse_elem))
    }
    overall_se_dict = {"rmse_se": rmse_se_overall, "mape_se": mape_se_overall}

    # PRINT METRICS
    print(
        "Metrics:\n"
        f"  Overall RMSE: {rmse_overall:.6f}  (SE: {overall_se_dict['rmse_se']:.6f})\n"
        f"  Overall MAPE: {mape_overall:.6f}  (SE: {overall_se_dict['mape_se']:.6f})\n"
        f"  R^2:          {r2_score:.6f}"
    )
    # Elementwise preview (first 8 elements)
    preview_k = min(8, len(elementwise_rmse_dict))
    if preview_k > 0:
        print("  Elementwise (first {k}) RMSE, MAPE, SEs:".format(k=preview_k))
        for i in range(preview_k):
            e = f"element_{i}"
            print(
                f"    {e}: RMSE={elementwise_rmse_dict[e]:.6f} "
                f"(SE={elementwise_se_dict[e]['rmse_se']:.6f}), "
                f"MAPE={elementwise_mape_dict[e]:.6f} "
                f"(SE={elementwise_se_dict[e]['mape_se']:.6f})"
            )

    return (
        rmse_overall,
        mape_overall,
        elementwise_rmse_dict,
        elementwise_mape_dict,
        r2_score,
        overall_se_dict,
        elementwise_se_dict,
    )
