import numpy as np
import torch
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_descriptor_space(
    model,
    train_x,
    valid_x,
    n_train_max=1000,
    n_valid_max=500,
    seed=0,
):
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def subsample(X, n_max, seed):
        rng = np.random.default_rng(seed)
        if len(X) > n_max:
            idx = rng.choice(len(X), n_max, replace=False)
            return X[idx]
        return X

    X_M = model.x_M.detach().cpu().numpy()

    X_train = np.concatenate([to_numpy(x) for x in train_x], axis=0)
    X_valid = np.concatenate([to_numpy(x) for x in valid_x], axis=0)

    X_train = subsample(X_train, n_train_max, seed)
    X_valid = subsample(X_valid, n_valid_max, seed + 1)

    X = np.vstack([X_train, X_valid, X_M])

    labels = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_valid), dtype=int),
        2 * np.ones(len(X_M), dtype=int),
    ])

    X_scaled = StandardScaler().fit_transform(X)

    n_pca = min(30, X_scaled.shape[1], X_scaled.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X_scaled)

    perplexity = min(30, max(2, (len(X_pca) - 1) // 3))

    X_emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X_pca)

    fig = go.Figure()

    for label_id, name, size, symbol, opacity in [
        (0, "train atoms", 5, "circle", 0.45),
        (1, "valid atoms", 7, "circle", 0.75),
        (2, "x_M inducing atoms", 12, "x", 1.0),
    ]:
        mask = labels == label_id
        fig.add_trace(
            go.Scatter(
                x=X_emb[mask, 0],
                y=X_emb[mask, 1],
                mode="markers",
                name=name,
                marker=dict(
                    size=size,
                    symbol=symbol,
                    opacity=opacity,
                ),
            )
        )

    fig.update_layout(
        width=800,
        height=650,
        title="Descriptor space: train / valid / inducing points (t-SNE)",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        font=dict(family="Arial", size=18),
        legend=dict(x=0.02, y=0.98),
    )

    fig.show()

    return fig, X_emb, labels