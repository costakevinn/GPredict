# utils.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Plot GP (prior or posterior)
# -----------------------------


def plot_gp(X_train=None, y_train=None, dy_train=None,
            X_test=None, mu=None, cov=None,
            title="Gaussian Process", filename="gp_plot.png"):
    """
    Plots a Gaussian Process with:
    - Posterior mean (mu)
    - 1-sigma region (fill_between)
    - Observational data with error bars (dy)
    Always saves the plot to file; does NOT display interactively.

    Args:
        X_train: np.ndarray, training inputs, shape (n_train,1)
        y_train: np.ndarray, training targets, shape (n_train,)
        dy_train: np.ndarray, observational uncertainties, shape (n_train,)
        X_test: np.ndarray, test/prediction points, shape (n_test,1)
        mu: np.ndarray, posterior mean at X_test
        cov: np.ndarray, posterior covariance at X_test
        title: str, plot title
        filename: str, output file path
    """
    plt.figure(figsize=(10,6))
    
    # GP posterior: mean + 1-sigma region
    if mu is not None and cov is not None and X_test is not None:
        std = np.sqrt(np.diag(cov))
        plt.fill_between(X_test.flatten(),
                         mu - std,
                         mu + std,
                         color='lightblue', alpha=0.5, label="1-sigma interval")
        plt.plot(X_test, mu, color='blue', lw=2, label='Mean')

    # Observational data with error bars
    if X_train is not None and y_train is not None:
        if dy_train is not None:
            plt.errorbar(X_train.flatten(), y_train, yerr=dy_train,
                         fmt='o', color='red', capsize=3, label='Observations')
        else:
            plt.scatter(X_train, y_train, color='red', s=50, label='Observations')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    # Always save the plot
    plt.savefig(filename)
    plt.close()



# -----------------------------
# 2. Save GP results to file
# -----------------------------
def save_results(X_test, mu, cov, filename="results.txt"):
    """
    Saves GP predictions to a text file with columns: x, mean, std.

    Args:
        X_test: np.ndarray, test points, shape (n_test,1)
        mu: np.ndarray, posterior mean, shape (n_test,)
        cov: np.ndarray, posterior covariance, shape (n_test,n_test)
        filename: str, output file
    """
    std = np.sqrt(np.diag(cov))
    data = np.column_stack((X_test.flatten(), mu, std))
    header = "x\tmu\tstd"
    np.savetxt(filename, data, header=header, fmt="%.6f", delimiter="\t")

# -----------------------------
# 3. Plot prior GP
# -----------------------------
def plot_prior(X_test, mu_prior, cov_prior, samples=None, title="GP Prior"):
    """
    Convenience function to plot prior GP.

    Args:
        X_test: np.ndarray, test points
        mu_prior: mean vector at X_test
        cov_prior: covariance matrix at X_test
        samples: optional, array of samples
        title: str
    """
    plot_gp(X_train=None, y_train=None, X_test=X_test, mu=mu_prior, cov=cov_prior,
            samples=samples, title=title)
