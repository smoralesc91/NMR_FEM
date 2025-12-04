import numpy as np
import pandas as pd
import scipy.optimize as opt
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
from scipy.optimize import fsolve
import openturns as ot
from fenics import *

__version__ = "1.0"

def NMR_Functions(__version__):
    pass

NMR_Functions.__version__ = __version__

########################################################################
# Normalize results data
def normalize_results(data):
    '''
    Return: norm_data
    '''
    vector = np.array(data)
    min_val = np.min(vector)
    max_val = np.max(vector)
    return (vector - min_val) / (max_val - min_val)

########################################################################
# Fit data
def T2star_fit(time_data, data):
    '''
    Return: t2star[0][1]
    '''
    def exp_function(x, a, b):
        return a * np.exp(-x / b)
    t2star, _ = opt.curve_fit(exp_function, time_data, data)
    return t2star[1]

########################################################################
# Calculating conventional T2star
def T2star_conventional(T2B, rho, SVratio):
    '''
    Return: T_2star
    '''
    if T2B <= 0:
        return 1. / (rho * SVratio)
    return 1. / ((1. / T2B) + rho * SVratio)

########################################################################
# Calculating analytical surface area and volume ratio
def SV_ratio_analytical(radius, aspect_ratio):
    '''
    Return: S, V, SVratio
    '''
    if aspect_ratio == 1:
        S = 4. * np.pi * radius**2
        V = 4. * np.pi * radius**3 / 3.
    else:
        a = radius
        b = c = radius * aspect_ratio
        S = 4. * np.pi * (((a * b)**1.6 + (a * c)**1.6 + (b * c)**1.6) / 3.)**(1. / 1.6)
        V = 4. * np.pi * a * b * c / 3.
            
    return S, V, S/V
    
########################################################################
# Calculating magnetic saturation (Curie's law)
# Global constants
AVOGADRO_NUMBER = 6.0220e23
H_PLANCK = 6.626e-34
K_BOLTZMANN = 1.380e-23
GAMMA = 267.5e6
    
# Fluid properties
FLUID_PROPERTIES = {
    'water': {'number_hydrogen': 2., 'mol_weight': 18.0153e-3, 'density': 9.97e2},
    'oil': {'number_hydrogen': 12., 'mol_weight': 72.151e-3, 'density': 6.26e2},
    'gas': {'number_hydrogen': 4., 'mol_weight': 16.04e-3, 'density': 6.56e-1}
    }
    
def mag_sat(B_0, Temp, fluid):
    if B_0 == 1 or Temp == 1 or fluid is None:
        return 1
        
    assert fluid in FLUID_PROPERTIES, "Fluid must be 'water', 'oil', or 'gas'"
        
    properties = FLUID_PROPERTIES[fluid]
    number_hydrogen = properties['number_hydrogen']
    mol_weight = properties['mol_weight']
    density = properties['density']
        
    proton_density = (number_hydrogen * AVOGADRO_NUMBER * density) / mol_weight
    m_s = (proton_density * B_0 * (GAMMA**2) * (H_PLANCK**2)) / (4. * K_BOLTZMANN * Temp)
    return m_s

########################################################################
# Mesh statistics
def mesh_statistics(mesh, T2B, rho):
    dim = mesh.topology().dim()
    if dim == 1:
        r2 = Expression('x[0]*x[0]', degree=3)
        volume_mesh = assemble(4 * np.pi * r2 * dx(mesh))
        surface_mesh = assemble(4 * np.pi * r2 * ds(mesh))
    else:
        volume_mesh = assemble(Constant(1) * dx(mesh))
        surface_mesh = assemble(Constant(1) * ds(mesh))
    
    surface_volume_ratio = surface_mesh / volume_mesh
        
    mesh_data = [
        ["hmin", f"{mesh.hmin():.4e}"],
        ["hmax", f"{mesh.hmax():.4e}"],
        ["num. cells", mesh.num_cells()],
        ["num. edges", mesh.num_edges()],
        ["num. entities 0d", mesh.num_entities(0)],
        ["num. entities 1d", mesh.num_entities(1)],
        ["num. entities 2d", mesh.num_entities(2)],
        ["num. entities 3d", mesh.num_entities(3)],
        ["num. faces", mesh.num_faces()],
        ["num. facets", mesh.num_facets()],
        ["num. vertices", mesh.num_vertices()],
        ["Mesh Volume", f"{volume_mesh:.4e} [m^3]"],
        ["Mesh Surface area", f"{surface_mesh:.4e} [m^2]"],
        ["Mesh Surface to Volume ratio", f"{surface_volume_ratio:.4e} [m^-1]"],
        ["T2star conventional", f"{T2star_conventional(T2B, rho, surface_volume_ratio):.4e} [s]"]
    ]
        
    print(tabulate(mesh_data, headers=["Mesh statistics", ""], tablefmt="github"))

########################################################################
# Brownstein-Tarr number
def BrownsteinTarr_number(r, rho, D, text=True):
    '''
    Return: kappa, kappa_regime
    '''
    kappa = (r * rho) / D
    if kappa <= 1:
        kappa_regime = 'Fast diffusion'
    elif kappa <= 10:
        kappa_regime = 'Intermediate diffusion'
    else:
        kappa_regime = 'Slow diffusion'

    if text:
        table = [["Pore [m]", "Regime", "Kappa"],
                 [f"{r:.2e}", kappa_regime, f"{kappa:.4f}"]]
        print(tabulate(table, headers="firstrow", tablefmt="github"))

    return kappa, kappa_regime

########################################################################
def compute_error_metrics(
    y_true,
    y_pred,
    *,
    tol: float = 1e-12,
    # pointwise relative error type
    relative: str | None = "mape",   # "mape" | "smape" | None
    baseline: str = "true",          # denominator in MAPE: "true" (|y_true|) or "pred" (|y_pred|)
    # normalized RMSE
    nrmse: str | None = None,        # None | "range" | "mean"
    # weighting
    weights = None,                  # array-like or None
    t = None,                        # time vector (non-uniform) -> trapezoidal weighting
    # output
    table: bool = False,             # prints table if desired
    digits: int = 6                  # decimals for printing
) -> dict:
    
    """Calculates a comprehensive set of error metrics between two signals.

    This function compares a 'predicted' signal (y_pred) to a 'true'
    signal (y_true). It automatically handles non-finite (NaN, Inf) values
    by filtering them out. It also supports optional weighting, either
    from a non-uniform time vector 't' (using the trapezoidal rule)
    or from a direct 'weights' array.

    Args:
        y_true (array-like): The ground truth or reference signal.
        y_pred (array-like): The predicted or computed signal to compare.
        tol (float, optional): A small tolerance value to prevent division by
            zero in relative metrics. Defaults to 1e-12.
        relative (str | None, optional): Specifies the type of pointwise
            relative error to calculate: "mape" (Mean Absolute Percentage Error)
            or "smape" (Symmetric Mean Absolute Percentage Error).
            Defaults to "mape".
        baseline (str, optional): If `relative="mape"`, determines the
            denominator: "true" (uses `|y_true|`) or "pred" (uses `|y_pred|`).
            Defaults to "true".
        nrmse (str | None, optional): Specifies the normalization for the
            Root Mean Squared Error: "range" (normalizes by `max(yt) - min(yt)`)
            or "mean" (normalizes by `mean(abs(yt))`). Defaults to None.
        weights (array-like | None, optional): An array of weights for each
            data point. If provided, all metrics will be weighted.
            Defaults to None.
        t (array-like | None, optional): A vector of time points, potentially
            non-uniform. If provided, weights are computed using the
            trapezoidal rule (integration weights). This overrides `weights`.
            Defaults to None.
        table (bool, optional): If True, prints a formatted table of the
            resulting metrics to the console. Defaults to False.
        digits (int, optional): The number of decimal digits to use when
            printing the table. Defaults to 6.

    Raises:
        ValueError: If inputs have mismatched shapes, if no finite data
            is found, if 't' is not strictly increasing, if 'weights'
            are invalid (negative, sum to zero), or if 'relative'
            or 'nrmse' keys are unrecognized.

    Returns:
        dict: A dictionary where keys are metric names (str) and values
        are the computed error values (float or int).

        Key metrics include:
        - count_used (int): Number of finite data points used.
        - MAE (float): Mean Absolute Error.
        - MSE (float): Mean Squared Error.
        - RMSE (float): Root Mean Squared Error.
        - L1_norm (float): L1 norm of the error (sum of absolute errors).
        - L2_norm (float): L2 norm of the error (sqrt of sum of squared errors).
        - Linf_norm (float): L-infinity norm (maximum absolute error).
        - RelL2 (float): L2 norm of the error normalized by the L2 norm of y_true.
        - R2 (float): R-squared (Coefficient of Determination).
        - MAPE_percent (float): Mean Absolute Percentage Error (if requested).
        - sMAPE_percent (float): Symmetric MAPE (if requested).
        - NRMSE_range/NRMSE_mean (float): Normalized RMSE (if requested).
    """
    
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("Input signals must have the same shape.")

    # filter non-finite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        raise ValueError("No finite data in y_true/y_pred.")
    yt = y_true[mask]
    yp = y_pred[mask]
    e  = yp - yt

    # weights
    if t is not None:
        t = np.asarray(t, dtype=float).ravel()
        if t.shape != y_true.shape:
            raise ValueError("t must have the same shape as y_true/y_pred.")
        t = t[mask]
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("t must be strictly increasing.")
        w = np.empty_like(t)
        w[1:-1] = 0.5*(dt[:-1] + dt[1:])
        w[0]  = 0.5*dt[0]
        w[-1] = 0.5*dt[-1]
        w = w / np.sum(w)
    elif weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != y_true.shape:
            raise ValueError("weights must have the same shape as y_true/y_pred.")
        w = w[mask]
        if np.any(w < 0):
            raise ValueError("weights must not contain negative values.")
        s = w.sum()
        if s == 0:
            raise ValueError("Sum of weights is zero.")
        w = w / s
    else:
        w = None

    # weighted average and "norm" helpers
    def wmean(z):
        return float(np.sum(w*z)) if w is not None else float(np.mean(z))

    def wsum_abs(z):
        return float(np.sum(np.abs(z)*w)*len(z)) if w is not None else float(np.sum(np.abs(z)))

    def wsum_sq(z):
        return float(np.sum((z*z)*w)*len(z)) if w is not None else float(np.sum(z*z))

    # basic metrics
    mae  = wmean(np.abs(e))
    mse  = wmean(e*e)
    rmse = np.sqrt(mse)

    # error norms
    l1_norm   = wsum_abs(e)
    l2_norm   = np.sqrt(wsum_sq(e))
    linf_norm = float(np.max(np.abs(e)))

    # L2 relative error
    denom_l2  = np.sqrt(wsum_sq(yt))
    rel_l2    = l2_norm / (denom_l2 + tol)

    # MAPE / sMAPE
    mape_percent = None
    smape_percent = None
    if relative is not None:
        rl = relative.lower()
        if rl == "mape":
            den = np.abs(yt) if baseline.lower() == "true" else np.abs(yp)
            mape_percent = wmean(np.abs(e) / (den + tol)) * 100.0
        elif rl == "smape":
            smape_percent = wmean(2.0*np.abs(e) / (np.abs(yt) + np.abs(yp) + tol)) * 100.0
        else:
            raise ValueError("relative must be 'mape', 'smape', or None.")

    # R^2
    yt_mean = wmean(yt)
    ss_res  = wsum_sq(yt - yp)
    ss_tot  = wsum_sq(yt - yt_mean)
    r2 = 1.0 - ss_res / (ss_tot + tol)

    # NRMSE
    nrmse_value = None
    nrmse_key = None
    if nrmse is not None:
        key = nrmse.lower()
        if key == "range":
            span = float(np.max(yt) - np.min(yt))
            nrmse_value = rmse / (span + tol)
            nrmse_key = "NRMSE_range"
        elif key == "mean":
            mean_abs = float(np.mean(np.abs(yt)))
            nrmse_value = rmse / (mean_abs + tol)
            nrmse_key = "NRMSE_mean"
        else:
            raise ValueError("nrmse must be None, 'range', or 'mean'.")

    # output
    metrics = {
        "count_used": int(yt.size),
        "MAE": mae,
        "AbsErrorMean": mae,  # explicit alias
        "MSE": mse,
        "RMSE": rmse,
        "L1_norm": l1_norm,
        "L2_norm": l2_norm,
        "Linf_norm": linf_norm,
        "RelL2": rel_l2,
        "R2": r2,
    }
    if mape_percent is not None:
        metrics["MAPE_percent"] = mape_percent
        metrics["RelErrorMean_percent"] = mape_percent  # compatibility alias
    if smape_percent is not None:
        metrics["sMAPE_percent"] = smape_percent
    if nrmse_value is not None:
        metrics[nrmse_key] = nrmse_value

    if table:
        try:
            from tabulate import tabulate
            rows = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    rows.append([k, f"{v:.{digits}e}"])
                else:
                    rows.append([k, v])
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="github"))
        except Exception:
            print("\n--- Error metrics ---")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"{k:<22}: {v:.{digits}e}")
                else:
                    print(f"{k:<22}: {v}")
            print("---------------------")

    return metrics

########################################################################
# Function for >1 peak
def maxT2_ilt(x, y, threshold=0.01):
    '''
    Return: max_y, x_position
    '''
    peaks, _ = find_peaks(y, height=threshold * np.max(y))
    if len(peaks) > 1:
        sorted_peaks = sorted(peaks, key=lambda i: y[i], reverse=True)[:2]
        max_y = [round(y[i], 8) for i in sorted_peaks]
        x_position = [round(x[i], 8) for i in sorted_peaks]
    else:
        index_max_y = np.argmax(y)
        max_y = [round(y[index_max_y], 8)]
        x_position = [round(x[index_max_y], 8)]
    return max_y, x_position

########################################################################
# Function for 1 peak
def maxT2_ilt_1peak(x, y, threshold=0.01):
    '''
    Return: max_y, x_position
    '''
    peaks, _ = find_peaks(y, height=threshold * np.max(y))
    if len(peaks) > 1:
        index_max_y = sorted(peaks, key=lambda i: y[i], reverse=True)[0]
    else:
        index_max_y = np.argmax(y)
    max_y = round(y[index_max_y], 8)
    x_position = round(x[index_max_y], 8)
    return max_y, x_position

########################################################################
# Kernel PDF estimation
def OT_kernel_pdf(data, data_sampling=100):
    '''
    Return: x, y
    '''
    data_ot = ot.Sample([[value] for value in data])
    kernel = ot.Epanechnikov()
    kernelSmoothing = ot.KernelSmoothing(kernel)
    fittedDistribution = kernelSmoothing.build(data_ot)
    x = np.linspace(np.min(data), np.max(data), data_sampling)
    y = np.array([fittedDistribution.computePDF([xi]) for xi in x])
    return x, y

########################################################################
# PLOTTING FUNCTIONS
# Plot single graph
def plot_single_graph(xdata=None, ydata=None, label='data', title='figure', xlabel='x', ylabel='y', xscale='linear',
                      marker='', linestyle='solid', color=None, xlim=None, ylim=None,
                      figsize=(20, 4), savefig=None):
    
    assert xscale in ['linear', 'log'], "xscale must be 'linear' or 'log'"
    
    plt.figure(figsize=figsize)
    plt.plot(xdata, ydata, marker=marker, linestyle=linestyle, color=color, label=label)
    plt.xscale(xscale)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

########################################################################
# Plot dual graphs
def plot_dual_graphs(xdata1=None, ydata1=None, label1='data1', title1='figure1', xlabel1='x', ylabel1='y', xscale1='linear',
                     xdata2=None, ydata2=None, label2='data2', title2='figure2', xlabel2='x', ylabel2='y', xscale2='linear',
                     marker1='', linestyle1='solid', marker2='', linestyle2='solid', color1=None, color2=None,
                     xlim1=None, ylim1=None, xlim2=None, ylim2=None,
                     figsize=(20, 4), savefig=None):

    assert xscale1 in ['linear', 'log'], "xscale1 must be 'linear' or 'log'"
    assert xscale2 in ['linear', 'log'], "xscale2 must be 'linear' or 'log'"
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].plot(xdata1, ydata1, marker=marker1, linestyle=linestyle1, color=color1, label=label1)
    axs[0].set_xscale(xscale1)
    if xlim1:
        axs[0].set_xlim(xlim1)
    if ylim1:
        axs[0].set_ylim(ylim1)
    axs[0].legend()
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel1)
    axs[0].set_ylabel(ylabel1)
    axs[0].grid(True, which='both')

    axs[1].plot(xdata2, ydata2, marker=marker2, linestyle=linestyle2, color=color2, label=label2)
    axs[1].set_xscale(xscale2)
    if xlim2:
        axs[1].set_xlim(xlim2)
    if ylim2:
        axs[1].set_ylim(ylim2)
    axs[1].legend()
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel2)
    axs[1].set_ylabel(ylabel2)
    axs[1].grid(True, which='both')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

########################################################################
# Plot triple graphs
def plot_triple_graphs(xdata1=None, ydata1=None, label1='data1', title1='figure1', xlabel1='x', ylabel1='y', xscale1='linear',
                       xdata2=None, ydata2=None, label2='data2', title2='figure2', xlabel2='x', ylabel2='y', xscale2='linear',
                       xdata3=None, ydata3=None, label3='data3', title3='figure3', xlabel3='x', ylabel3='y', xscale3='linear',
                       marker1='', linestyle1='solid', marker2='', linestyle2='solid', marker3='', linestyle3='solid',
                       color1=None, color2=None, color3=None, xlim1=None, ylim1=None, xlim2=None, ylim2=None, xlim3=None, ylim3=None,
                       figsize=(20, 4), savefig=None):

    assert xscale1 in ['linear', 'log'], "xscale1 must be 'linear' or 'log'"
    assert xscale2 in ['linear', 'log'], "xscale2 must be 'linear' or 'log'"
    assert xscale3 in ['linear', 'log'], "xscale3 must be 'linear' or 'log'"

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    axs[0].plot(xdata1, ydata1, marker=marker1, linestyle=linestyle1, color=color1, label=label1)
    axs[0].set_xscale(xscale1)
    if xlim1:
        axs[0].set_xlim(xlim1)
    if ylim1:
        axs[0].set_ylim(ylim1)
    axs[0].legend()
    axs[0].set_title(title1)
    axs[0].set_xlabel(xlabel1)
    axs[0].set_ylabel(ylabel1)
    axs[0].grid(True, which='both')

    axs[1].plot(xdata2, ydata2, marker=marker2, linestyle=linestyle2, color=color2, label=label2)
    axs[1].set_xscale(xscale2)
    if xlim2:
        axs[1].set_xlim(xlim2)
    if ylim2:
        axs[1].set_ylim(ylim2)
    axs[1].legend()
    axs[1].set_title(title2)
    axs[1].set_xlabel(xlabel2)
    axs[1].set_ylabel(ylabel2)
    axs[1].grid(True, which='both')

    axs[2].plot(xdata3, ydata3, marker=marker3, linestyle=linestyle3, color=color3, label=label3)
    axs[2].set_xscale(xscale3)
    if xlim3:
        axs[2].set_xlim(xlim3)
    if ylim3:
        axs[2].set_ylim(ylim3)
    axs[2].legend()
    axs[2].set_title(title3)
    axs[2].set_xlabel(xlabel3)
    axs[2].set_ylabel(ylabel3)
    axs[2].grid(True, which='both')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

######################################################################
def profile_to_csv(
    profiles_dict,
    r_grid,
    output_filename='spatial_profiles.csv'
):
    """
    Saves the calculated spatial magnetization profiles to a CSV file.

    This function takes the dictionary of profiles and the radial grid array,
    assembles them into a pandas DataFrame, and exports them to a CSV file.
    The time-based columns are formatted to include ' s' (seconds) for clarity.

    Parameters
    ----------
    profiles_dict : dict
        A dictionary where keys are the time points (float) and
        values are the corresponding 1D magnetization profile arrays (np.ndarray).
    r_grid : np.ndarray
        A 1D array containing the radial coordinates (the r-axis).
    output_filename : str, optional
        The desired name for the output CSV file.
        Defaults to 'spatial_profiles.csv'.

    Returns
    -------
    None
        This function does not return a value; its side effect is
        saving a file to disk.
    """
    
    print(f"\nAssembling data to save to {output_filename}...")

    # 1. Create a pandas DataFrame from the profiles dictionary.
    df = pd.DataFrame(profiles_dict)

    # 2. Rename columns to add " s" for clarity.
    column_rename_map = {t: f"{t} s" for t in df.columns}
    df = df.rename(columns=column_rename_map)

    # 3. Insert the radial grid as the first column (index 0)
    df.insert(0, 'r_grid', r_grid)

    # 4. Save the DataFrame to the specified CSV file.
    df.to_csv(output_filename, index=False)

    print(f"Success! Results have been saved to: {output_filename}")
    print("The CSV file contains the following columns:")
    print(list(df.columns))

##################################################################