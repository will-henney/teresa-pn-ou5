"""
Utilities that chatgpt has helped me to write

regression_with_ci() is a reimplementation of seaborn.regplot()
but that also returns the regression fit parameters


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def regression_with_ci(
    x, y,
    order=1,
    ci=95,
    n_boot=1000,
    robust=False,
    scatter=True,
    plot=True,
    seed=None,
    ax=None,
    **kwargs
):
    """
    Polynomial regression with bootstrap confidence intervals and optional robust fitting.

    Works with Seaborn 0.13.2 and Python 3.11.

    Parameters:
        x, y     : array-like inputs
        order    : polynomial degree (default: 1 = linear)
        ci       : confidence level (default: 95)
        n_boot   : number of bootstrap resamples (default: 1000)
        robust   : use robust regression via statsmodels.RLM (only if order == 1)
        scatter  : plot scatter points
        plot     : show matplotlib plot
        seed     : random seed for reproducibility
        **kwargs : passed to line plot (e.g. color, linestyle)

    Returns:
        dict with keys:
            - coefficients: fitted model coefficients
            - x_fit: x values used for plotting
            - y_pred: predicted fit
            - ci_lower: lower confidence band
            - ci_upper: upper confidence band
    """
    x = np.asarray(x)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    
    x_fit = np.linspace(np.min(x), np.max(x), 100)

    def design_matrix(x_vals, order):
        return np.vander(x_vals, N=order + 1, increasing=False)

    def fit_model(x_sample, y_sample):
        if robust and order == 1:
            X = design_matrix(x_sample, order)
            model = sm.RLM(y_sample, X, M=sm.robust.norms.HuberT())
            results = model.fit()
            return results.params
        else:
            return np.polyfit(x_sample, y_sample, order)

    def predict(x_vals, coeffs):
        return np.polyval(coeffs, x_vals)

    # Fit on full data
    coeffs = fit_model(x, y)
    y_pred = predict(x_fit, coeffs)

    # Bootstrap
    boot_preds = np.empty((n_boot, len(x_fit)))
    for i in range(n_boot):
        indices = rng.integers(0, len(x), size=len(x))
        x_sample = x[indices]
        y_sample = y[indices]
        try:
            c = fit_model(x_sample, y_sample)
            boot_preds[i] = predict(x_fit, c)
        except Exception:
            boot_preds[i] = np.nan  # Handle potential RLM failures

    # Compute confidence interval
    ci_lower = np.nanpercentile(boot_preds, (100 - ci) / 2, axis=0)
    ci_upper = np.nanpercentile(boot_preds, 100 - (100 - ci) / 2, axis=0)

    # Plot
    if plot:
        if ax is None:
            ax = plt.gca()
        if scatter:
            sns.scatterplot(x=x, y=y, ax=ax, **kwargs)
        ax.plot(x_fit, y_pred, label="Robust fit" if robust else "Fit", **kwargs)
        ax.fill_between(x_fit, ci_lower, ci_upper, color='gray', alpha=0.3, label=f"{ci}% CI")
        ax.legend()

    return {
        "coefficients": coeffs,
        "x_fit": x_fit,
        "y_pred": y_pred,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

