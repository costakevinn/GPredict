# examples.py
# GP examples for GPredict
# Generates synthetic data with realistic observational errors
# Saves: observations, posterior plots, results

import numpy as np
import os
from gp import fit_gp, predict_gp, sample_gp, create_prediction_grid
from utils import plot_gp, save_results
from kernels import rbf_kernel, matern_kernel
from means import linear_mean, constant_mean


# -------------------------------
# Helper: generate observational data
# -------------------------------
def generate_observational_data(true_func, t_values, instrument_error=0.05,
                                trend_coeff=0.01, noise_sigma=0.02):
    """
    Generate y_data and dy_data with realistic observational errors
    """
    trend_error = trend_coeff * t_values
    gaussian_noise = np.random.normal(0, noise_sigma, size=len(t_values))
    dy_data = np.abs(instrument_error + trend_error + gaussian_noise)
    y_data = true_func(t_values).flatten() + np.random.normal(0, dy_data)
    return y_data, dy_data

# -------------------------------
# Example functions (true functions)
# -------------------------------
def sinusoidal_func(t):
    return np.sin(t)

def linear_func(t):
    return 2.0 * t + 1.0

def quadratic_func(t):
    return -5 * t**2 + 1.0 * t

# -------------------------------
# Generic runner
# -------------------------------
def run_example(true_func, mean_func, mean_params, kernel_func, kernel_params,
                t_values, instrument_error, trend_coeff, noise_sigma, name,
                min_border=1.0, max_border=1.0, n_pred=200, n_samples=3):
    
    # Prepare directories
    data_path = f"data/{name}"
    plots_path = f"plots/{name}"
    results_path = f"results/{name}"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Generate noisy observations
    y_data, dy_data = generate_observational_data(true_func, t_values,
                                                  instrument_error, trend_coeff, noise_sigma)
    
    # Save observations
    obs_file = f"{data_path}/observations.txt"
    with open(obs_file, "w") as f:
        f.write("t\t y\t dy\n")
        for t, y, dy in zip(t_values, y_data, dy_data):
            f.write(f"{t:.4f}\t{y:.4f}\t{dy:.4f}\n")
    print(f"Observational data saved in {obs_file}")

    # Prediction grid
    X_test = create_prediction_grid(t_values.reshape(-1,1), min_border, max_border, n_pred)

    # --- PLOT PRIOR ---
    # Vetor de média do prior
    mu_prior = mean_func(X_test, **mean_params).flatten()  # usa a função de média

    # Matriz de covariância do prior
    n_test = X_test.shape[0]
    cov_prior = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in range(n_test):
            cov_prior[i,j] = kernel_func(X_test[i], X_test[j], **kernel_params)

    # Plot prior
    plot_gp(X_test=X_test, mu=mu_prior, cov=cov_prior,
            title=f"{name} GP Prior",
            filename=f"{plots_path}/prior.png")


    # Fit GP
    L, alpha, mean_train = fit_gp(t_values.reshape(-1,1), y_data, dy_data,
                                  mean_func=mean_func, mean_params=mean_params,
                                  kernel_func=kernel_func, kernel_params=kernel_params)

    # Posterior prediction
    mu, cov = predict_gp(t_values.reshape(-1,1), X_test, L, alpha,
                        mean_func=mean_func, mean_params=mean_params,
                        kernel_func=kernel_func, kernel_params=kernel_params)

    # Garantir que mu seja 1D antes de plotar
    mu = mu.flatten()  # <-- adicione isto aqui

    # Samples from posterior
    samples = sample_gp(mu, cov, n_samples=n_samples)

    # Plot posterior
    plot_gp(X_train=t_values, y_train=y_data, dy_train=dy_data,
            X_test=X_test, mu=mu, cov=cov,
            title=f"{name} GP Posterior",
            filename=f"{plots_path}/posterior.png")


    # Save results
    save_results(X_test, mu, cov, filename=f"{results_path}/results.txt")

    print(f"Example '{name}' complete! Check {plots_path} and {results_path}")

# -------------------------------
# Specific runners
# -------------------------------
def run_sinusoidal_example():
    t_values = np.linspace(0, 10, 20)
    mean_params = {"w":[0.0], "b":0.0}
    kernel_params = {"lengthscale": 1.0, "sigma_f": 1.0}
    run_example(sinusoidal_func, linear_mean, mean_params, rbf_kernel, kernel_params,
                t_values, instrument_error=0.05, trend_coeff=0.02, noise_sigma=0.05,
                name="sinusoidal")

def run_linear_example():
    t_values = np.linspace(0, 10, 20)
    mean_params = {"w":[0.0], "b":0.0}
    kernel_params = {"lengthscale": 1.0, "sigma_f": 1.0}
    run_example(linear_func, linear_mean, mean_params, rbf_kernel, kernel_params,
                t_values, instrument_error=0.3, trend_coeff=0.02, noise_sigma=0.05,
                name="linear")

def run_quadratic_example():
    t_values = np.linspace(0, 5, 20)
    mean_params = {"w":[0.0], "b":0.0}
    kernel_params = {"lengthscale": 0.5, "sigma_f": 10}
    run_example(quadratic_func, linear_mean, mean_params, rbf_kernel, kernel_params,
                t_values, instrument_error=5, trend_coeff=0.02, noise_sigma=0.03,
                name="quadratic")
