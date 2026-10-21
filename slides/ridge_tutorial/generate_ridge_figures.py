import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#============================================================

#Configuration

#============================================================

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
    # “”“Save the current figure and close it.”””
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

#============================================================

#Figure 1: Overfitting

#============================================================

def figure_overfitting():
    rng = np.random.default_rng(RANDOM_STATE)

    X = np.sort(rng.uniform(-3, 3, 25))
    y = np.sin(X) + rng.normal(0, 0.18, size=X.shape)
    X_grid = np.linspace(-3.2, 3.2, 500)
    models = {
        "Linear model (underfitting example)": make_pipeline(
            PolynomialFeatures(degree=1, include_bias=False),
            LinearRegression()
        ),
        "Polynomial model (degree 4)": make_pipeline(
            PolynomialFeatures(degree=4, include_bias=False),
            LinearRegression()
        ),
        "Polynomial model (degree 15)": make_pipeline(
            PolynomialFeatures(degree=15, include_bias=False),
            LinearRegression()
        ),
    }
    plt.figure()
    plt.yscale('log')
    plt.ylim(1e-40, 1e2)
    plt.yticks([1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10])
    plt.scatter(
        X, y,
        s=45,
        label="Training observations",
        zorder=3
    )
    for name, model in models.items():
        model.fit(X.reshape(-1, 1), y)
        prediction = model.predict(X_grid.reshape(-1, 1))
        plt.plot(X_grid, prediction, linewidth=2, label=name)
    plt.title("Model Complexity and Overfitting")
    plt.xlabel("Input feature x")
    plt.ylabel("Target y")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure("01_overfitting.png")

#============================================================

#Figure 2: Ridge regularization

#============================================================

def figure_ridge_idea():
    rng = np.random.default_rng(RANDOM_STATE)

    X = rng.normal(size=(80, 1))
    y = 2.0 + 3.0 * X[:, 0] + rng.normal(0, 1.4, size=80)
    X_grid = np.linspace(-2.8, 2.8, 300).reshape(-1, 1)
    alphas = [0.0, 1.0, 20.0, 100.0]
    plt.figure()
    plt.scatter(
        X[:, 0], y,
        s=35,
        alpha=0.65,
        label="Training observations"
    )
    for alpha in alphas:
        if alpha == 0:
            model = LinearRegression()
            label = "OLS (alpha = 0)"
        else:
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=alpha)
            )
            label = f"Ridge (alpha = {alpha:g})"
        model.fit(X, y)
        prediction = model.predict(X_grid)
        plt.plot(
            X_grid[:, 0],
            prediction,
            linewidth=2,
            label=label
        )
    plt.title("How Regularization Changes the Fitted Line")
    plt.xlabel("Input feature x")
    plt.ylabel("Target y")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure("02_ridge_idea.png")

#============================================================

#Figure 3: Coefficient shrinkage

#============================================================

def figure_coefficient_shrinkage():
    rng = np.random.default_rng(RANDOM_STATE)

    n_samples = 100
    X = rng.normal(size=(n_samples, 4))
    # Create correlated predictors.
    X[:, 1] = 0.85 * X[:, 0] + 0.3 * X[:, 1]
    X[:, 2] = 0.75 * X[:, 0] + 0.4 * X[:, 2]
    true_coef = np.array([3.0, 2.5, -2.0, 0.8])
    y = X @ true_coef + rng.normal(0, 1.2, size=n_samples)
    # Standardize features so coefficient shrinkage is comparable.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    alphas = np.logspace(-3, 3, 100)
    coefficients = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, y)
        coefficients.append(model.coef_)
    coefficients = np.asarray(coefficients)
    plt.figure()
    for j in range(X.shape[1]):
        plt.plot(
            alphas,
            coefficients[:, j],
            linewidth=2,
            label=f"Feature {j + 1}"
        )
    plt.xscale("log")
    plt.axhline(0, linewidth=1)
    plt.title("Ridge Coefficients vs. Regularization Strength")
    plt.xlabel("Regularization parameter alpha (log scale)")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure("03_coefficient_shrinkage.png")

#============================================================

# Figure 4: Geometric interpretation of Ridge

#============================================================

def figure_ridge_geometry():
    # Illustrative two-coefficient example.
    beta1 = np.linspace(-3, 5, 400)
    beta2 = np.linspace(-5, 3, 400)

    B1, B2 = np.meshgrid(beta1, beta2)
    # Illustrative least-squares objective centered away
    # from the origin.
    loss = (B1 - 3.0) ** 2 + 1.5 * (B2 + 2.0) ** 2
    # Ridge penalty contours.
    penalty = B1 ** 2 + B2 ** 2
    plt.figure()
    loss_levels = [2, 5, 10, 20, 35]
    penalty_levels = [2, 5, 10, 15, 20]
    cs1 = plt.contour(
        B1, B2, loss,
        levels=loss_levels,
        linestyles="solid"
    )
    plt.clabel(cs1, inline=True, fontsize=8)
    cs2 = plt.contour(
        B1, B2, penalty,
        levels=penalty_levels,
        linestyles="dashed"
    )
    plt.clabel(cs2, inline=True, fontsize=8)
    plt.scatter(
        [3.0], [-2.0],
        marker="x",
        s=100,
        linewidths=3,
        label="OLS solution"
    )
    plt.scatter(
        [0.0], [0.0],
        marker="o",
        s=45,
        label="Origin"
    )
    plt.title("Ridge Regression: Loss and Penalty Contours")
    plt.xlabel(r"Coefficient $\beta_1$")
    plt.ylabel(r"Coefficient $\beta_2$")
    plt.legend()
    plt.grid(alpha=0.2)
    save_figure("04_ridge_geometry.png")

#============================================================

# Figure 5: Illustrative bias-variance trade-off

#============================================================

def figure_bias_variance():
    # These are illustrative curves, not estimates from a
    # particular dataset.
    alpha = np.logspace(-4, 4, 200)
    log_alpha = np.log10(alpha)

    # Illustrative trend: bias increases with regularization.
    bias_squared = 0.15 + 0.09 * (log_alpha + 4) ** 1.5
    # Illustrative trend: variance decreases with regularization.
    variance = 4.5 / (1.0 + np.exp(1.2 * (log_alpha - 0.2)))
    total_error = bias_squared + variance
    plt.figure()
    plt.plot(
        alpha,
        bias_squared,
        linewidth=2,
        label="Illustrative squared bias"
    )
    plt.plot(
        alpha,
        variance,
        linewidth=2,
        label="Illustrative variance"
    )
    plt.plot(
        alpha,
        total_error,
        linewidth=3,
        label="Illustrative total error"
    )
    best_index = np.argmin(total_error)
    plt.axvline(
        alpha[best_index],
        linestyle="--",
        linewidth=1.5,
        label="Illustrative minimum"
    )
    plt.xscale("log")
    plt.title("The Bias–Variance Trade-off")
    plt.xlabel("Regularization parameter alpha (log scale)")
    plt.ylabel("Illustrative error contribution")
    plt.legend()
    plt.grid(alpha=0.25)
    save_figure("05_bias_variance.png")

#============================================================

# Main

#============================================================

if __name__ == "__main__":
    figure_overfitting()
    figure_ridge_idea()
    figure_coefficient_shrinkage()
    figure_ridge_geometry()
    figure_bias_variance()

    print("\nAll figures generated successfully.")