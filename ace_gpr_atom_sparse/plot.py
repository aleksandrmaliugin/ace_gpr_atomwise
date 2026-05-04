import numpy as np
import torch
import plotly.graph_objects as go


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def predict_structures(model, X_structures, return_std=True):
    model.eval()

    with torch.no_grad():
        if return_std:
            pred, std = model.predict_uncertainty(X_structures)
            return to_numpy(pred), to_numpy(std)
        else:
            pred = model(X_structures)
            return to_numpy(pred), None


def mae_metric_np(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def rmse_metric_np(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def plot_results(
    model,
    train_x,
    train_y,
    valid_x,
    valid_y,
    save_plot: bool = False,
    filename: str = "gpr_accuracy.pdf",
):
    model.eval()

    train_y_np = to_numpy(train_y)
    valid_y_np = to_numpy(valid_y)

    train_pred, train_std = predict_structures(
        model,
        train_x,
        return_std=True,
    )

    valid_pred, valid_std = predict_structures(
        model,
        valid_x,
        return_std=True,
    )

    mae_train = mae_metric_np(train_pred, train_y_np)
    mae_valid = mae_metric_np(valid_pred, valid_y_np)

    rmse_train = rmse_metric_np(train_pred, train_y_np)
    rmse_valid = rmse_metric_np(valid_pred, valid_y_np)

    xy_min = np.min(
        np.concatenate([train_pred, valid_pred, train_y_np, valid_y_np])
    )
    xy_max = np.max(
        np.concatenate([train_pred, valid_pred, train_y_np, valid_y_np])
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.linspace(xy_min, xy_max, 10),
            y=np.linspace(xy_min, xy_max, 10),
            mode="lines",
            line=dict(color="grey"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=train_pred,
            y=train_y_np,
            mode="markers",
            name=(
                f"Train: RMSE = {rmse_train * 1000:.2f} meV, "
                f"MAE = {mae_train * 1000:.2f} meV"
            ),
            marker=dict(size=12),
            error_x=dict(
                type="data",
                array=train_std,
                visible=True,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=valid_pred,
            y=valid_y_np,
            mode="markers",
            name=(
                f"Valid: RMSE = {rmse_valid * 1000:.2f} meV, "
                f"MAE = {mae_valid * 1000:.2f} meV"
            ),
            marker=dict(size=12),
            error_x=dict(
                type="data",
                array=valid_std,
                visible=True,
            ),
        )
    )

    fig.update_xaxes(
        title="E<sub>model</sub>, eV",
        title_font=dict(size=25),
        tickfont=dict(size=22),
        automargin=True,
    )

    fig.update_yaxes(
        title="E<sub>DFT</sub>, eV",
        title_font=dict(size=25),
        tickfont=dict(size=22),
        automargin=True,
    )

    fig.update_layout(
        width=750,
        height=750,
        margin=dict(r=50, t=50, pad=4),
        font=dict(family="Arial", size=20),
        legend=dict(x=0.02, y=0.98),
    )

    if save_plot:
        fig.write_image(filename)

    fig.show()

    metrics = {
        "rmse_train": rmse_train,
        "rmse_valid": rmse_valid,
        "mae_train": mae_train,
        "mae_valid": mae_valid,
    }

    return fig, metrics