import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
"""
============================================================

Configuration

============================================================
"""
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 180,
})

def save_figure(filename):
    """Save the current figure and close it."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
"""
============================================================

Figure 1: Overfitting

============================================================
"""
def figure_overfitting():
    rng = np.random.default_rng(RANDOM_STATE)

    X = np.sort(rng.uniform(-3, 3, 25))
    y = np.sin(X) + rng.normal(0, 0.18, size=X.shape)
    X_grid = np.linspace(-3.2, 3.2, 500)
    models = {
        "Linear model": make_pipeline(
            PolynomialFeatures(degree=1, include_bias=False),
            LinearRegression()
        ),
        "Polynomial degree 4": make_pipeline(
            PolynomialFeatures(degree=4, include_bias=False),
            LinearRegression()
        ),
        "Polynomial degree 15": make_pipeline(
            PolynomialFeatures(degree=15, include_bias=False),
            LinearRegression()
        ),
    }
    plt.figure()
    plt.scatter(
        X, y,
        s=45,
        label="Training observations",
        zorder=3
    )
    for name, model in models.items():
        model.fit(X.reshape(-1, 1), y)
        y_grid = model.predict(X_grid.reshape(-1, 1))
        plt.plot(
            X_grid,
            y_grid,
            linewidth=2,
            label=name
        )
    plt.title("Model Complexity and Overfitting")
    plt.xlabel("Input feature x")
    plt.ylabel("Target y")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure("01_overfitting.png")
"""
============================================================

Figure 2: Lasso coefficient shrinkage paths

============================================================
"""
def figure_coefficient_shrinkage():
    rng = np.random.default_rng(RANDOM_STATE)

    n_samples = 120
    n_features = 8
    X = rng.normal(size=(n_samples, n_features))
    # Introduce correlation between some features.
    X[:, 1] = 0.85 * X[:, 0] + 0.35 * X[:, 1]
    X[:, 3] = 0.75 * X[:, 2] + 0.45 * X[:, 3]
    true_coef = np.array([
        3.0, 0.0, -2.5, 0.0,
        1.8, 0.0, 0.0, 0.8
    ])
    y = X @ true_coef + rng.normal(0, 1.0, n_samples)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-3, 1.2, 120)
    coefficient_paths = []
    for alpha in alphas:
        model = Lasso(
            alpha=alpha,
            max_iter=50000,
            tol=1e-7
        )
        model.fit(X_scaled, y)
        coefficient_paths.append(model.coef_)
    coefficient_paths = np.asarray(coefficient_paths)
    plt.figure()
    for j in range(n_features):
        plt.plot(
            alphas,
            coefficient_paths[:, j],
            linewidth=2,
            label=f"Feature {j + 1}"
        )
    plt.xscale("log")
    plt.axhline(0, linewidth=1)
    plt.title("Lasso Coefficient Paths")
    plt.xlabel("Regularization parameter alpha (log scale)")
    plt.ylabel("Coefficient value")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    save_figure("02_coefficient_shrinkage.png")
"""
============================================================

Figure 3: Geometric interpretation of Lasso

============================================================
"""
def figure_lasso_geometry():
    beta1 = np.linspace(-1.0, 5.0, 500)
    beta2 = np.linspace(-4.0, 2.0, 500)

    B1, B2 = np.meshgrid(beta1, beta2)
    # Illustrative least-squares loss contours.
    loss = (
        (B1 - 3.0) ** 2
        + 1.5 * (B2 + 2.0) ** 2
    )
    plt.figure()
    loss_levels = [1, 3, 6, 10, 16, 24]
    contours = plt.contour(
        B1,
        B2,
        loss,
        levels=loss_levels,
        linewidths=1.5
    )
    plt.clabel(contours, inline=True, fontsize=8)
    # L1 constraint boundaries:
    # |beta1| + |beta2| = t
    for t in [1, 2, 3, 4, 5]:
        points = np.array([
            [t, 0],
            [0, t],
            [-t, 0],
            [0, -t],
            [t, 0],
        ])
        plt.plot(
            points[:, 0],
            points[:, 1],
            linestyle="--",
            linewidth=1.2,
            label=f"L1 boundary t={t}" if t == 2 else None
        )
    plt.scatter(
        [3.0],
        [-2.0],
        marker="x",
        s=100,
        linewidths=3,
        label="Illustrative OLS solution"
    )
    plt.scatter(
        [0],
        [0],
        marker="o",
        s=40,
        label="Origin"
    )
    plt.title("Lasso Geometry: Loss Contours and L1 Diamonds")
    plt.xlabel(r"Coefficient $\beta_1$")
    plt.ylabel(r"Coefficient $\beta_2$")
    plt.legend()
    plt.grid(alpha=0.2)
    save_figure("03_lasso_geometry.png")
"""
============================================================

Figure 4: Feature selection demonstration

============================================================
"""
def figure_feature_selection():
    # Generate data with only a few genuinely useful features.
    X, y, true_coef = make_regression(
    n_samples=180,
    n_features=20,
    n_informative=5,
    noise=15,
    coef=True,
    random_state=RANDOM_STATE
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE
    )
    # Standardization is fitted only on the training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    alphas = [0.01, 0.1, 1.0, 10.0, 30.0]
    selected_counts = []
    train_errors = []
    test_errors = []
    for alpha in alphas:
        model = Lasso(
            alpha=alpha,
            max_iter=50000
        )
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        selected_counts.append(
            np.count_nonzero(model.coef_)
        )
        train_errors.append(
            mean_squared_error(y_train, y_train_pred)
        )
        test_errors.append(
            mean_squared_error(y_test, y_test_pred)
        )
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(
        alphas,
        selected_counts,
        marker="o",
        linewidth=2,
        label="Nonzero coefficients"
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Regularization parameter alpha")
    ax1.set_ylabel("Number of selected features")
    ax1.set_title("Lasso Feature Selection and Prediction Error")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        alphas,
        train_errors,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Training MSE"
    )
    ax2.plot(
        alphas,
        test_errors,
        marker="^",
        linestyle="-.",
        linewidth=2,
        label="Test MSE"
    )
    ax2.set_ylabel("Mean squared error")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center"
    )
    save_figure("04_lasso_feature_selection.png")

"""
============================================================

Main

============================================================
"""
if __name__ == "__main__":
    figure_overfitting()
    figure_coefficient_shrinkage()
    figure_lasso_geometry()
    figure_feature_selection()

    print("\nAll Lasso figures generated successfully.")