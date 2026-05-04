import torch
from torch.utils.data import random_split
from copy import deepcopy


def train_valid_split(
        dataset: torch.utils.data.Dataset,
        train_fraction: float= 0.8,
        seed: int = 42,
    ):
    train_size = int(train_fraction * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, valid_dataset

def get_tensors_from_subset(subset):
    idx = list(subset.indices)

    X = [subset.dataset.X[i] for i in idx]
    y = subset.dataset.y[idx]

    return X, y

    
def rmse_metric(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def mae_metric(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def train_sparse_atomic_gpr(
    train_x,
    train_y,
    valid_x,
    valid_y,
    optimizer,
    scheduler,
    model,
    n_epochs=500,
    device=torch.device("cpu"),
    dtype=torch.float64,
    model_path="best_sparse_atomic_gpr.pt",
    min_lr=1e-5,
):
    model = model.to(device)

    train_x = [torch.as_tensor(x, dtype=dtype, device=device) for x in train_x]
    valid_x = [torch.as_tensor(x, dtype=dtype, device=device) for x in valid_x]

    train_y = torch.as_tensor(train_y, dtype=dtype, device=device)
    valid_y = torch.as_tensor(valid_y, dtype=dtype, device=device)

    history = {
        "rmse_train": [],
        "rmse_valid": [],
        "mae_train": [],
        "mae_valid": [],
        "sigma2": [],
        "lengthscale_mean": [],
        "outputscale": [],
        "lr": [],
    }

    best_rmse_valid = float("inf")

    for epoch in range(n_epochs):
        model.train()

        optimizer.zero_grad()
        loss, _, _ = model.training_loss(train_x, train_y)
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            model.fit_c(
                train_x,
                train_y,
                build_uncertainty=False,
            )

            pred_train = model(train_x)
            pred_valid = model(valid_x)

            rmse_train = rmse_metric(pred_train, train_y)
            rmse_valid = rmse_metric(pred_valid, valid_y)

            mae_train = mae_metric(pred_train, train_y)
            mae_valid = mae_metric(pred_valid, valid_y)

        rmse_train_val = rmse_train.item()
        rmse_valid_val = rmse_valid.item()
        mae_train_val = mae_train.item()
        mae_valid_val = mae_valid.item()
        lr_val = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(rmse_valid_val)
            else:
                scheduler.step()

            lr_val = optimizer.param_groups[0]["lr"]

        history["rmse_train"].append(rmse_train_val)
        history["rmse_valid"].append(rmse_valid_val)
        history["mae_train"].append(mae_train_val)
        history["mae_valid"].append(mae_valid_val)
        history["sigma2"].append(model.sigma2.item())
        history["lengthscale_mean"].append(model.lengthscale.mean().item())
        history["outputscale"].append(model.outputscale.item())
        history["lr"].append(lr_val)

        if rmse_valid_val < best_rmse_valid:
            best_rmse_valid = rmse_valid_val

            with torch.no_grad():
                model.fit_c(
                    train_x,
                    train_y,
                    build_uncertainty=True,
                )
                model.save(model_path)

        if epoch % 50 == 0:
            print(
                f"Iter {epoch+1}/{n_epochs} "
                f"RMSE train: {rmse_train_val:.6f} "
                f"RMSE valid: {rmse_valid_val:.6f} "
                f"MAE train: {mae_train_val:.6f} "
                f"MAE valid: {mae_valid_val:.6f} "
                f"best RMSE valid: {best_rmse_valid:.6f} "
                f"sigma2: {model.sigma2.item():.3e} "
                f"ls_mean: {model.lengthscale.mean().item():.3e} "
                f"outputscale: {model.outputscale.item():.3e} "
                f"lr: {lr_val:.3e}"
            )

        if lr_val < min_lr:
            break

    return history, best_rmse_valid