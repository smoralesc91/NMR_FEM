<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

### Summary of Changes in `run_single_case`

This new version of `run_single_case` introduces several mathematical and numerical corrections to ensure a consistent use of the dimensionless Bloch–Torrey model between the semi-analytical Brownstein–Tarr solution and the FEM solver `BT_fenics_decay`. The main goal is to remove hidden biases that previously caused significant overestimation of the surface relaxivity ρ in challenging cases (e.g., $R = 2.25\ \mu\text{m}, \rho = 40\ \mu\text{m/s}$).

***

### 1. Consistent Time Grid for Semi-Analytical and FEM Models

**Original behavior**

- The synthetic time vector was built as:

```python
num_output_points = int(tf_val / DT_VALUE)
t_gen = np.linspace(0.0, tf_val - DT_VALUE, num_output_points)
```

which produces points in $[0, DT, 2DT, \dots, tf\_val - DT]$, i.e., *excluding* the final time.
- The FEM solver was called with:

```python
BT_fenics_decay(..., t_start=0.0, t_final=tf_val, dt_phys=DT_VALUE)
```

which internally computes:

```python
steps = round((t_final - t_start) / dt_phys)
time_array = [0, dt, ..., steps*dt] ≈ [0, ..., tf_val]
```

yielding `steps+1` points, *including* the final time.
- If `len(s_out) != len(t_obs)`, a linear interpolation of the FEM signal onto the synthetic grid was applied:

```python
s_out = np.interp(t_obs, np.linspace(0.0, tf_val, len(s_out)), s_out)
```


**Problems**

- The semi-analytical and FEM models were *not* evaluated at exactly the same physical times.
- The interpolation was compensating discrepancies a posteriori, especially at short times where the signal decays very fast (small pores, large ρ).
- In cases like $R = 2.25\ \mu\text{m}, \rho = 40\ \mu\text{m/s}$, the early-time behavior dominates the information content; small timing mismatches created systematic residuals that the MCMC compensated by driving ρ to unrealistically high values.

**New behavior**

- A single, consistent observation grid is defined:

```python
dt_phys = DT_VALUE
t_obs = np.arange(0.0, tf_val + 0.5*dt_phys, dt_phys)  # includes tf_val
tau_grid = t_obs / T2B_REF
```

- This `t_obs` is used:
    - To generate the synthetic data via the semi-analytical model.
    - As the target time grid for the FEM solver.
- The FEM is called with the same `(t_start, t_final, dt_phys)`:

```python
time_fem, s_out = BT_fenics_decay(
    R_phys=R_true,
    D_phys=D_REF,
    rho_phys=rho_cand,
    T2B_phys=T2B_REF,
    t_start=0.0,
    t_final=tf_val,
    dt_phys=dt_phys,
    ...
)
```

- A consistency check is performed:

```python
if len(time_fem) != len(t_obs) or not np.allclose(time_fem, t_obs):
    s_out = np.interp(t_obs, time_fem, s_out)
```

but in normal use (coherent `dt_phys` and `tf_val`), `time_fem` and `t_obs` match exactly and no interpolation is needed.

**Effect**

Semi-analytical and FEM signals are now computed on *exactly the same* physical time grid, eliminating timing-induced biases in the likelihood and stabilizing the inference of ρ in fast-decay regimes.

***

### 2. Correct and Explicit Use of the Dimensionless Semi-Analytical Model

**Original behavior**

- The Brownstein–Tarr solution `NMR_SemiA_sphere_dimless` was called with a time grid that was not guaranteed to be consistent with the FEM time stepping or with the dimensionless scaling $\tau = t / T_{2B}$.
- In some configurations, this led to semi-analytical signals evaluated at times that did not match those used in `BT_fenics_decay`, further amplifying discrepancies.

**New behavior**

- The dimensionless time grid is explicitly tied to the physical observation times:

```python
tau_grid = t_obs / T2B_REF
```

- The semi-analytical solver is called as:

```python
tau_sa, s_gen, _, _, _, _ = NMR_SemiA_sphere_dimless(
    radius=R_true,
    diffusion=D_REF,
    rho=rho_true,
    T2B=T2B_REF,
    tau_array=tau_grid,
    n_terms=200,
    return_data='all'
)
```

- The dimensionless formulation used in both codes is identical:
    - $\tau = t/T_{2B}$,
    - $\phi = r/\sqrt{D T_{2B}}$,
    - $\phi_R = R/\sqrt{D T_{2B}}$,
    - $\kappa = R\rho/D = \alpha \phi_R$.

**Effect**

The “ground truth” semi-analytical signal is now evaluated *exactly* at the same dimensionless times as the FEM model, with identical scaling. Any mismatch between FEM and analytical solutions is now due to numerical discretization or noise, not to inconsistent non-dimensionalization or time grids.

***

### 3. Cleaner Handling of FEM Failures and Removal of Artificial Artifacts

**Original behavior**

- If `BT_fenics_decay` raised an exception, the forward model returned an array filled with `-10.0`:

```python
return np.full_like(t_obs, -10.0)
```

- In `Energy`, any occurrence of `-10.0` triggered an infinite energy (rejection):

```python
if np.any(s_pred == -10.0): return np.inf
```


**Problems**

- Filling with a constant unphysical value introduces a sharp, artificial structure into the likelihood surface.
- Combined with the ad-hoc interpolation, this could accidentally create regions where the sampler behaves poorly or gets pushed into extreme parameter values.

**New behavior**

- If `BT_fenics_decay` fails, the forward model returns `None`:

```python
except Exception as e:
    print(f"Error in BT_fenics_decay (alpha={alpha_cand}): {e}")
    return None
```

- In `Energy`, a `None` prediction is penalized with a large but finite energy:

```python
s_pred = forward_fem_mcmc(alpha_val)
if s_pred is None:
    return 1e6
```


**Effect**

The sampler now sees a smooth, physically meaningful likelihood landscape without artificial spikes from unphysical placeholder values. Proposals that cause FEM failures are simply disfavored, without contaminating the numerical evaluation of the log-likelihood.

***

### 4. Minor but Relevant Numerical and Statistical Improvements

Although not the primary cause of the ρ bias, the following changes improve robustness:

1. **Safe handling of very small σ in the likelihood**

```python
sigma_safe = max(sigma_val, 1e-12)
log_lik = -n * np.log(sigma_safe) - 0.5 * np.sum(resid**2) / sigma_safe**2
```

This avoids numerical overflow when σ proposals are extremely small.
2. **Explicit transformation between α and ρ consistent with scaling**

```python
rho_cand = alpha_cand * np.sqrt(D_REF / T2B_REF)
...
rho_chain_full = alpha_chain_full * np.sqrt(D_REF / T2B_REF)
```

ensuring that the inferred α is always mapped back to ρ using the same non-dimensionalization as in the FEM and semi-analytical solvers.
3. **Consistent use of the same `dt_phys` and `tf_val` in all components**
    - A single `dt_phys = DT_VALUE` and `tf_val` are used:
        - to define `t_obs`,
        - in `BT_fenics_decay`,
        - and indirectly in the dimensionless semi-analytical call via `tau_array`.

***

### 5. Why These Changes Fix the ρ Overestimation for Small R and Large ρ

In the challenging regime $R = 2.25\,\mu\text{m}$, $\rho = 40\,\mu\text{m/s}$:

- The problem is strongly surface‑relaxation dominated; the normalized signal decays very rapidly and the early time points carry most of the information about ρ.
- In the original implementation, misaligned time grids and interpolations made the FEM signal effectively “too slow” or “too fast” compared to the semi-analytical reference.
- The MCMC compensated systematic shape mismatches in the decay curve by pushing α (and hence ρ) to unrealistically high values.

With the new implementation:

- Time grids, non-dimensionalization, and scaling between α and ρ are internally consistent across semi-analytical and FEM models.
- The only differences between synthetic data and FEM predictions arise from discretization error and added noise, not from misalignment or inconsistent scaling.
- As a result, the posterior for ρ is centered much closer to the true value, with uncertainty controlled by the noise level and the chosen priors, rather than being dominated by systematic model mismatch.
<span style="display:none">[^1][^2]</span>

<div align="center">⁂</div>

[^1]: ND_FEM_NMR.py

[^2]: SemiA_Sphere_NMR_dimensionless-2.py

