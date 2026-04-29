import numpy as np
import torch
import gpytorch
import plotly.graph_objects as go


def predict_structures(model, X_structures):
    model.eval()
    model.likelihood.eval()

    with model.covar_module.register_query_structures(X_structures) as ids:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = model(ids)

    return pred.mean.detach().cpu().numpy(), pred.stddev.detach().cpu().numpy()


def plot_results(
    model,
    train_x,
    train_y,
    valid_x,
    valid_y,
    save_plot: bool = False,
):
    model.eval()
    model.likelihood.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    train_y_np = train_y.detach().cpu().numpy()
    valid_y_np = valid_y.detach().cpu().numpy()

    # train prediction: use train ids
    train_ids = model.train_inputs[0].to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_train = model(train_ids)

    train_pred = pred_train.mean.detach().cpu().numpy()
    train_std = pred_train.stddev.detach().cpu().numpy()

    # valid prediction: register valid structures
    valid_x = [x.to(device=device, dtype=dtype) for x in valid_x]

    with model.covar_module.register_query_structures(valid_x) as valid_ids:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_valid = model(valid_ids)

    valid_pred = pred_valid.mean.detach().cpu().numpy()
    valid_std = pred_valid.stddev.detach().cpu().numpy()

    MAE_train = np.mean(np.abs(train_pred - train_y_np))
    MAE_valid = np.mean(np.abs(valid_pred - valid_y_np))

    xy_min = np.min(np.concatenate((train_pred, valid_pred, train_y_np, valid_y_np)))
    xy_max = np.max(np.concatenate((train_pred, valid_pred, train_y_np, valid_y_np)))

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
            name=f"MAE<sub>train</sub> = {MAE_train * 1000:.3f} meV",
            marker=dict(size=15, color="#636EFA"),
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
            name=f"MAE<sub>valid</sub> = {MAE_valid * 1000:.3f} meV",
            marker=dict(size=15, color="#FFA15A"),
            error_x=dict(
                type="data",
                array=valid_std,
                visible=True,
            ),
        )
    )

    fig.update_xaxes(
        title="E<sub>model</sub>, eV",
        ticklabelstep=2,
        title_font=dict(size=25),
        tickfont=dict(size=25),
        automargin=True,
    )

    fig.update_yaxes(
        title="E<sub>DFT</sub>, eV",
        ticklabelstep=2,
        automargin=True,
        title_font=dict(size=25),
        tickfont=dict(size=25),
    )

    fig.update_layout(
        width=800,
        height=600,
        margin=dict(r=50, t=50, pad=4),
        font=dict(family="Arial", size=23),
    )

    fig.update_legends(x=0.77, y=0.5)

    if save_plot:
        fig.write_image("gpr_accuracy.pdf")

    fig.show()

    return fig