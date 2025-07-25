# NMR_FEM
This repository provides the computational tools for numerically simulating the Bloch-Torrey equation using the Finite Element Method (FEM) and solving its associated inverse problem. 

Primarily written in Python, it leverages FEniCS for forward solutions and NumPy alongside SciPy for post-processing and the inversion of the Fredholm integral equation of the first kind via a Regularized Non-Negative Least Squares (RNNLS) algorithm. 

The codebase is designed for full reproducibility, including scripts for mesh generation, FEM discretization, time-stepping solvers, inverse modeling, and data visualization.
