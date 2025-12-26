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

from typing import Union, Dict, Optional

def compute_error_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *,
    t: Optional[Union[np.ndarray, list]] = None,
    tol: float = 1e-12,
    table: bool = False,
    digits: int = 6
) -> Dict[str, Union[int, float]]:
    """
    Computes comprehensive error metrics for physical signal analysis with optional
    time-domain weighting for continuous signals.
    
    This function calculates absolute and relative error metrics suitable for comparing
    continuous signals, particularly in applications like NMR spectroscopy, where 
    signals may be sampled at non-uniform time intervals. The metrics include 
    functional norms (L1, L2, L∞) and statistical measures (MAE, RMSE, R²).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth signal values. Can be 1D or 2D array (will be flattened).
        Must contain finite numerical values.
    y_pred : array-like
        Predicted or reconstructed signal values. Must have same shape as y_true.
    t : array-like, optional
        Time vector corresponding to signal samples. Required for weighted metrics 
        with non-uniform sampling. When provided, metrics are computed using 
        trapezoidal integration weights. Must be strictly increasing. 
        If None, uniform sampling is assumed.
    tol : float, default=1e-12
        Numerical tolerance to prevent division by zero in relative error calculations.
    table : bool, default=False
        If True, prints a formatted table summarizing all computed metrics.
    digits : int, default=6
        Number of decimal digits displayed in the output table (when table=True).
    
    Returns
    -------
    dict
        Dictionary containing the following error metrics:
        
        **Basic Information**
        count : int
            Number of valid (finite) data points used for calculations.
        
        **Absolute Error Metrics**
        MAE : float
            Mean Absolute Error - average of absolute errors (weighted if t provided).
        RMSE : float
            Root Mean Square Error - square root of average squared errors.
        L1_norm : float
            L1 norm of the error - integral of absolute error over domain.
        L2_norm : float
            L2 norm of the error - square root of integral of squared error.
        Linf_norm : float
            L∞ norm (maximum norm) - maximum absolute error across all points.
        
        **Relative Error Metrics**
        Global_Rel_Error_Pct : float
            Global relative error percentage based on L1 norm:
            ||error||₁ / ||true_signal||₁ × 100%
        Rel_L2 : float
            Relative L2 error (dimensionless):
            ||error||₂ / ||true_signal||₂
        
        **Statistical Measure**
        R2 : float
            Coefficient of determination (R² score) - proportion of variance explained.
            Range: (-∞, 1], where 1 indicates perfect prediction.
    
    Notes
    -----
    1. **Weighting Logic**:
       - When `t` is provided, metrics use trapezoidal integration weights for 
         proper treatment of non-uniform sampling.
       - Integration weights (`w_int`) are used for computing norms (L1, L2).
       - Statistical weights (`w_stat = w_int / sum(w_int)`) are used for averages.
       - For uniform sampling (t=None), all points receive equal weight.
    
    2. **Error Norm Definitions**:
       - L1 norm: ∫|error(t)| dt (approximated by trapezoidal rule)
       - L2 norm: √[∫error(t)² dt] (approximated by trapezoidal rule)
       - L∞ norm: max(|error(t)|) (no weighting applied)
    
    3. **Relative Error Calculations**:
       - Global_Rel_Error_Pct: Compares integrated absolute error to integrated 
         absolute signal magnitude.
       - Rel_L2: Compares integrated squared error to integrated squared signal 
         magnitude.
    
    4. **R² Calculation**:
       - Uses weighted mean and sums when time weights are provided.
       - Standard unweighted formula for uniform sampling.
       - Formula: R² = 1 - SS_res / SS_tot, where SS_res is residual sum of squares
         and SS_tot is total sum of squares about the (weighted) mean.
    
    5. **Robustness Features**:
       - Automatically filters out non-finite values (NaN, inf).
       - Adds tolerance (`tol`) to denominators to avoid division by zero.
       - Handles edge cases with single or few data points.
       - Validates time vector monotonicity.
    
    Examples
    --------
    >>> # Basic usage with uniform sampling
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 1.9, 3.2, 3.8]
    >>> metrics = compute_error_metrics(y_true, y_pred)
    >>> print(f"MAE: {metrics['MAE']:.3f}, R²: {metrics['R2']:.3f}")
    
    >>> # With non-uniform time sampling
    >>> t = [0, 1, 3, 6]  # Non-uniform intervals
    >>> metrics = compute_error_metrics(y_true, y_pred, t=t)
    >>> print(f"L2 norm: {metrics['L2_norm']:.3f}")
    
    >>> # With table display
    >>> metrics = compute_error_metrics(y_true, y_pred, t=t, table=True)
    
    Raises
    ------
    ValueError
        - If y_true and y_pred have different shapes.
        - If no finite data is available after filtering.
        - If t is provided and contains non-increasing values.
        - If t length doesn't match number of valid data points.
    
    See Also
    --------
    numpy.trapz : Trapezoidal integration method.
    sklearn.metrics.mean_absolute_error : Similar MAE calculation.
    sklearn.metrics.r2_score : Similar R² calculation.
    """
    
    # ============================================================================
    # DATA VALIDATION AND PREPROCESSING
    # ============================================================================
    
    # Convert inputs to 1D float arrays and flatten to ensure consistent processing
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    
    # Validate array dimensions match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")
    
    # Create mask to exclude non-finite values (NaN, inf) from both arrays
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    # Check if any valid data remains
    if not np.any(mask):
        raise ValueError("No finite data points available for error computation.")
    
    # Apply mask to obtain clean datasets for analysis
    yt = y_true[mask]
    yp = y_pred[mask]
    
    # Compute error vector: prediction - ground truth
    e = yp - yt
    N = len(yt)
    
    # ============================================================================
    # WEIGHT CALCULATION FOR NON-UNIFORM SAMPLING
    # ============================================================================
    
    if t is not None:
        # Extract and mask time values corresponding to valid data points
        t = np.asarray(t, dtype=float).ravel()[mask]
        
        # Validate time vector length matches data
        if len(t) != N:
            raise ValueError(
                f"Time vector length ({len(t)}) must match number of valid "
                f"data points ({N})."
            )
        
        # Check for strictly increasing time values
        diffs = np.diff(t)
        if np.any(diffs <= 0):
            raise ValueError("Time vector t must contain strictly increasing values.")
        
        # Initialize integration weights array
        dt = np.zeros(N)
        
        # Handle edge cases based on number of points
        if N == 1:
            # Single point: unit weight for integration
            dt[0] = 1.0
        elif N == 2:
            # Two points: trapezoidal rule simplifies to equal halves
            dt[0] = 0.5 * diffs[0]
            dt[1] = 0.5 * diffs[0]
        else:
            # Three or more points: full trapezoidal integration weights
            # Interior points: average of adjacent intervals
            dt[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
            # Boundary points: half of first/last interval
            dt[0] = 0.5 * diffs[0]
            dt[-1] = 0.5 * diffs[-1]
        
        # Integration weights represent the dt for trapezoidal integration
        integration_weights = dt
        
        # Total domain span for normalization
        domain_span = np.sum(integration_weights)
        
        # Statistical weights for computing averages (normalized to sum to 1)
        stat_weights = integration_weights / domain_span
    else:
        # Uniform sampling case: equal weights for all points
        integration_weights = np.ones(N)
        stat_weights = np.ones(N) / N
    
    # ============================================================================
    # ABSOLUTE ERROR METRICS (FUNCTIONAL NORMS)
    # ============================================================================
    
    # L1 norm: ∫|error(t)| dt (trapezoidal approximation)
    # Represents the total absolute error integrated over the domain
    l1_norm = np.sum(np.abs(e) * integration_weights)
    
    # L2 norm squared: ∫error(t)² dt
    l2_sq = np.sum((e**2) * integration_weights)
    
    # L2 norm: √[∫error(t)² dt]
    # Represents the Euclidean distance between signals
    l2_norm = np.sqrt(l2_sq)
    
    # L∞ norm (maximum norm): max|error(t)|
    # Identifies the worst-case error point
    linf_norm = float(np.max(np.abs(e)))
    
    # ============================================================================
    # RELATIVE ERROR METRICS
    # ============================================================================
    
    # Denominator for L1-based relative error: ∫|true_signal(t)| dt
    den_l1 = np.sum(np.abs(yt) * integration_weights)
    
    # Global relative error (L1-based): ||error||₁ / ||true_signal||₁ × 100%
    # Provides percentage error relative to total signal magnitude
    rel_l1_percent = (l1_norm / (den_l1 + tol)) * 100.0
    
    # Denominator for L2-based relative error: √[∫true_signal(t)² dt]
    den_l2 = np.sqrt(np.sum((yt**2) * integration_weights))
    
    # Relative L2 error: ||error||₂ / ||true_signal||₂
    # Dimensionless measure of error relative to signal energy
    rel_l2 = l2_norm / (den_l2 + tol)
    
    # ============================================================================
    # STATISTICAL ERROR METRICS
    # ============================================================================
    
    # Mean Absolute Error (MAE): ∫|error(t)| w_stat dt
    # Weighted average of absolute errors
    mae = np.sum(np.abs(e) * stat_weights)
    
    # Mean Squared Error (MSE): ∫error(t)² w_stat dt
    mse = np.sum((e**2) * stat_weights)
    
    # Root Mean Square Error (RMSE): √MSE
    # Standard deviation of prediction errors
    rmse = np.sqrt(mse)
    
    # ============================================================================
    # COEFFICIENT OF DETERMINATION (R² SCORE)
    # ============================================================================
    
    # Weighted mean of ground truth signal
    yt_mean = np.sum(yt * stat_weights)
    
    # Residual sum of squares (weighted)
    ss_res = np.sum((yt - yp)**2 * stat_weights)
    
    # Total sum of squares (weighted)
    ss_tot = np.sum((yt - yt_mean)**2 * stat_weights)
    
    # R² = 1 - SS_res / SS_tot (with tolerance to avoid division by zero)
    r2 = 1.0 - ss_res / (ss_tot + tol)
    
    # ============================================================================
    # RESULTS COMPILATION
    # ============================================================================
    
    metrics = {
        # Basic information
        "count": int(N),
        
        # Statistical measure
        "R2": r2,
        
        # Absolute error metrics
        "MAE": mae,
        "RMSE": rmse,
        "L1_norm": l1_norm,
        "L2_norm": l2_norm,
        "Linf_norm": linf_norm,
        
        # Relative error metrics
        "Rel_L2": rel_l2,
        "Global_Rel_Error_Pct": rel_l1_percent,
    }
    
    # ============================================================================
    # OPTIONAL TABLE DISPLAY
    # ============================================================================
    
    if table:
        print("\n" + "=" * 60)
        print("ERROR METRICS SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<30} {'Value':>20}")
        print("-" * 60)
        
        # Display formatting based on metric type
        for key, value in metrics.items():
            if key == "count":
                # Integer formatting for count
                print(f"{key:<30} {int(value):>20}")
            elif key == "R2":
                # R² typically shown with 4-6 decimal places
                print(f"{key:<30} {value:>20.6f}")
            elif "Pct" in key:
                # Percentage formatting for relative errors
                print(f"{key:<30} {value:>20.4f} %")
            elif "norm" in key or key in ["MAE", "RMSE"]:
                # Scientific notation for norms and absolute errors
                print(f"{key:<30} {value:>20.6e}")
            else:
                # Default formatting for other metrics
                print(f"{key:<30} {value:>20.6f}")
        
        print("=" * 60)
        print(f"Total points analyzed: {N}")
        if t is not None:
            domain_length = t[-1] - t[0] if len(t) > 1 else 0
            print(f"Time domain span: {domain_length:.4f}")
    
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
