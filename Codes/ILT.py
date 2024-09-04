import numpy as np
import scipy.optimize as opt 
from Functions_NMR import normalize_results

"""Module to implement routines of numerical inverse 
Laplace tranform using Contin  algorithm [1]

[1] Provencher, S. (1982)

Original code was rewritten in python by caizkun.
"""

#Solver for Least Distance Programming (LDP) with constraint.
def ldp(G, h):
    m, n = G.shape
    if m != h.shape[0]:
        print ('\nError in ldp(): input G and h have different dimensions!')

    E = np.concatenate((G.T, h.reshape(1, m)))
    f = np.zeros(n+1)
    f[n] = 1.

    u, resnorm = opt.nnls(E, f)

    r = np.dot(E, u) - f

    if np.linalg.norm(r) == 0:
        print ('\nError in ldp(): solution is incompatible with inequality!')
    else:
        x = -r[0:-1]/r[-1]
    return x

#Inverse Laplace Transform
def ilt(t, F, bound, Nz, alpha, normed=False):
    if len(t) != len(F):
        print ('Error in ilt(): array t has different dimension from array F!')
    if len(F) < Nz:
        print ('Error in ilt(): Nz is expected smaller than the dimension of F!')

    h = np.log(bound[1]/bound[0])/(Nz - 1)     
    z = bound[0]*np.exp(np.arange(Nz)*h)       

    z_mesh, t_mesh = np.meshgrid(z, t)
    C = np.exp(-t_mesh*z_mesh)                 
    C[:, 0] /= 2.
    C[:, -1] /= 2.
    C *= h

    Nreg = Nz + 2
    R = np.zeros([Nreg, Nz])
    R[0, 0] = 1.
    R[-1, -1] = 1.
    R[1:-1, :] = -2*np.diag(np.ones(Nz)) + np.diag(np.ones(Nz-1), 1) \
        + np.diag(np.ones(Nz-1), -1)

    U, H, Z = np.linalg.svd(R, full_matrices=False)     
    Z = Z.T
    H = np.diag(H)

    Hinv = np.diag(1.0/np.diag(H))
    Q, S, W = np.linalg.svd(C.dot(Z).dot(Hinv), full_matrices=False)  
    W = W.T
    S = np.diag(S)

    Gamma = np.dot(Q.T, F)
    Sdiag = np.diag(S)
    Salpha = np.sqrt(Sdiag**2 + alpha**2)
    GammaTilde = Gamma*Sdiag/Salpha
    Stilde = np.diag(Salpha)
 
    Stilde_inv = np.diag(1.0/np.diag(Stilde))
    G = Z.dot(Hinv).dot(W).dot(Stilde_inv)
    B = -np.dot(G, GammaTilde)

    Xi = ldp(G, B)

    zf = np.dot(G, Xi + GammaTilde)
    f = zf/z

    F_restored = np.dot(C, zf)

    res_lsq = F - np.dot(C, zf)
    mean_res_lsq = np.mean(res_lsq)
    
    res_reg = np.dot(R, zf)
    mean_res_reg = np.mean(res_reg)

    if not normed:
        return z, f, mean_res_lsq, mean_res_reg, F_restored
    else:
        return z, normalize_results(f), mean_res_lsq, mean_res_reg, normalize_results(F_restored)