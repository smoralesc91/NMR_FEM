import numpy as np
import scipy.optimize as opt
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import openturns as ot
from fenics import *

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
# Absolute and relative global error, L2 norm and Infty norm
def error_estimation(true_data, measured_data, tol=1.e-10, table=False):
    '''
    Return: abs_error_glob, rel_error_glob, l2_norm, inf_norm
    '''
    abs_error_glob = np.mean(np.abs(measured_data - true_data))
    rel_error_glob = np.mean(np.abs(measured_data - true_data) / (np.abs(true_data) + tol)) * 100

    if np.size(true_data) > 1:
        l2_norm = np.linalg.norm(measured_data - true_data, ord=2)
        inf_norm = np.linalg.norm(measured_data - true_data, ord=np.inf)
    else:
        l2_norm = inf_norm = None

    if table:
        headers = ["Metric", "Value"]
        values = [
            ["Absolute Global Error", abs_error_glob],
            ["Relative Global Error (%)", rel_error_glob]
        ]
        if l2_norm is not None and inf_norm is not None:
            values.extend([
                ["L2 Norm", l2_norm],
                ["Infinity Norm", inf_norm]
            ])
        
        print(tabulate(values, headers, tablefmt="github"))

    return (abs_error_glob, rel_error_glob, l2_norm, inf_norm) if l2_norm is not None else (abs_error_glob, rel_error_glob)

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