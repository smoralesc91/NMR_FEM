import numpy as np
import scipy.optimize as opt
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Normalize results data
def normalize_results(data):
    '''
    Return: norm_data
    '''
    data_max = np.max(np.abs(data))
    norm_data = data/data_max
    return norm_data

# Fit data
def T2star_fit(time_data, data):
    '''
    Return: t2star[0][1]
    '''
    def exp_function(x, a, b):
        return a*np.exp(-x/b)
    t2star = opt.curve_fit(exp_function, time_data, data)
    return t2star[0][1]
    
# Calculating conventional T2star
def T2star_conventional(T2B, rho, SVratio):
    '''
    Return: T_2star
    '''
    if T2B <= 0:
        T_2star = 1. / (rho * SVratio)
    else:
        T_2star = 1. / ((1. / T2B) + rho * SVratio)
    return T_2star

# Calculating analytical surface area and volume ratio
def SV_ratio_analytical(radius, aspect_ratio):
    '''
    Return: S, V, SVratio
    '''
    if aspect_ratio == 1:
        S = 4. * np.pi * np.power(radius, 2)
        V = 4. * np.pi * np.power(radius, 3) / 3.
    else:
        a = radius
        b = radius*aspect_ratio
        c = radius*aspect_ratio
        S = 4. * np.pi * (((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3.) ** (1. / 1.6)
        V = 4. * np.pi * a * b * c / 3.
            
    SVratio = S / V
    return S, V, SVratio

# Brownstein-Tarr number
def BrownsteinTarr_number(r, rho, D, text=True):
    '''
    Return: kappa, kappa_regime
    '''
    kappa = (r * rho) / D
    if kappa <= 1:
        kappa_regime = 'Fast diffusion'
    elif 1 <= kappa <= 10:
        kappa_regime = 'Intermediate diffusion'
    else:  # kappa > 10
        kappa_regime = 'Slow diffusion'

    if text:
        table = [["Pore [m]", "Regime", "Kappa"],
                 ["{:.2e}".format(r), kappa_regime, "{:.4f}".format(kappa)]]
        print(tabulate(table, headers="firstrow", tablefmt="github"))

    return kappa, kappa_regime

# Absolute and relative global error, L2 norm and Infty norm
def error_estimation(true_data, measured_data, tol=1.e-10, table=False):
    '''
    Return: abs_error_glob, rel_error_glob, l2_norm, inf_norm
    '''
    abs_error_glob = np.mean(np.abs(measured_data-true_data))
    rel_error_glob = np.mean(np.abs(measured_data-true_data)/(np.abs(true_data)+tol))*100
    l2_norm = np.linalg.norm(measured_data-true_data, ord=2)
    inf_norm = np.linalg.norm(measured_data-true_data, ord=np.inf)
    
    if table:
        headers = ["Metric", "Value"]
        values = [
            ["Absolute Global Error", abs_error_glob],
            ["Relative Global Error (%)", rel_error_glob],
            ["L2 Norm", l2_norm],
            ["Infinity Norm", inf_norm]
        ]
        tab = tabulate(values, headers, tablefmt="github")
        print(tab)
    
    return abs_error_glob, rel_error_glob, l2_norm, inf_norm

def maxT2_ilt(x, y, threshold=0.01):
    '''
    Return: max_y, x_position
    '''
    peaks, _ = find_peaks(y, height=threshold*np.max(y))
    if len(peaks) > 1:
        sorted_peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
        max_y = [round(y[sorted_peaks[0]], 6), 
                 round(y[sorted_peaks[1]], 6)]
        x_position = [round(x[sorted_peaks[0]], 6), 
                      round(x[sorted_peaks[1]], 6)]
    else:
        index_max_y = np.argmax(y)
        max_y = [round(y[index_max_y], 6)]
        x_position = [round(x[index_max_y], 6)]
    return max_y, x_position

# Plot single graph
def plot_single_graph(xdata=None, ydata=None, label='data', title='figure',
                     figsize=(20, 4), savefig=None):
    
    plt.figure(figsize=figsize)
    plt.plot(xdata, ydata, label=label)
    plt.legend()
    plt.title(title)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

# Plot dual graphs
def plot_dual_graphs(xdata1=None, ydata1=None, label1='data1', title1='figure1',
                     xdata2=None, ydata2=None, label2='data2', title2='figure2',
                     figsize=(20, 4), savefig=None):

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].plot(xdata1, ydata1, label=label1)
    axs[0].legend()
    axs[0].set_title(title1)
    axs[0].grid(True)

    axs[1].plot(xdata2, ydata2, label=label2)
    axs[1].legend()
    axs[1].set_title(title2)
    axs[1].grid(True)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

# Plot tripe graphs
def plot_tripĺe_graphs(xdata1=None, ydata1=None, label1='data1', title1='figure1', xscale1='lineal',
                       xdata2=None, ydata2=None, label2='data2', title2='figure2', xscale2='lineal',
                       xdata3=None, ydata3=None, label3='data3', title3='figure3', xscale3='lineal',
                       figsize=(20,4), savefig=None):

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    if xscale1 == 'lineal':
        axs[0].plot(xdata1, ydata1, label=label1)
    elif xscale1 == 'log':
        axs[0].semilogx(xdata1, ydata1, label=label1)
    axs[0].legend()
    axs[0].set_title(title1)
    axs[0].grid(True, which='both')

    if xscale2 == 'lineal':
        axs[1].plot(xdata2, ydata2, label=label2)
    elif xscale2 == 'log':
        axs[1].semilogx(xdata2, ydata2, label=label2)
    axs[1].legend()
    axs[1].set_title(title2)
    axs[1].grid(True, which='both')

    if xscale3 == 'lineal':
        axs[2].plot(xdata3, ydata3, label=label3)
    elif xscale3 == 'log':
        axs[2].semilogx(xdata3, ydata3, label=label3)
    axs[2].legend()
    axs[2].set_title(title3)
    axs[2].grid(True, which='both')

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()