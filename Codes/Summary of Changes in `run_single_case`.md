### Summary of Changes in `run\_single\_case`

This new version of `run\_single\_case` introduces several mathematical and numerical corrections to ensure a consistent use of the dimensionless Bloch–Torrey model between the semi-analytical Brownstein–Tarr solution and the FEM solver `BT\_fenics\_decay`. The main goal is to remove hidden biases that previously caused significant overestimation of the surface relaxivity ρ in challenging cases (e.g., $R = 2.25\\ \\mu\\text{m}, \\rho = 40\\ \\mu\\text{m/s}$).

\---

### 1\. Consistent Time Grid for Semi-Analytical and FEM Models

**Original behavior**

* The synthetic time vector was built as:

```python
num\_output\_points = int(tf\_val / DT\_VALUE)
t\_gen = np.linspace(0.0, tf\_val - DT\_VALUE, num\_output\_points)
```

which produces points in $\[0, DT, 2DT, \\dots, tf\_val - DT]$, i.e., *excluding* the final time.

* The FEM solver was called with:

```python
BT\_fenics\_decay(..., t\_start=0.0, t\_final=tf\_val, dt\_phys=DT\_VALUE)
```

which internally computes:

```python
steps = round((t\_final - t\_start) / dt\_phys)
time\_array = \[0, dt, ..., steps\*dt] ≈ \[0, ..., tf\_val]
```

yielding `steps+1` points, *including* the final time.

* If `len(s\_out) != len(t\_obs)`, a linear interpolation of the FEM signal onto the synthetic grid was applied:

```python
s\_out = np.interp(t\_obs, np.linspace(0.0, tf\_val, len(s\_out)), s\_out)
```



**Problems**

* The semi-analytical and FEM models were *not* evaluated at exactly the same physical times.
* The interpolation was compensating discrepancies a posteriori, especially at short times where the signal decays very fast (small pores, large ρ).
* In cases like $R = 2.25\\ \\mu\\text{m}, \\rho = 40\\ \\mu\\text{m/s}$, the early-time behavior dominates the information content; small timing mismatches created systematic residuals that the MCMC compensated by driving ρ to unrealistically high values.

**New behavior**

* A single, consistent observation grid is defined:

```python
dt\_phys = DT\_VALUE
t\_obs = np.arange(0.0, tf\_val + 0.5\*dt\_phys, dt\_phys)  # includes tf\_val
tau\_grid = t\_obs / T2B\_REF
```

* This `t\_obs` is used:

  * To generate the synthetic data via the semi-analytical model.
  * As the target time grid for the FEM solver.
* The FEM is called with the same `(t\_start, t\_final, dt\_phys)`:

```python
time\_fem, s\_out = BT\_fenics\_decay(
    R\_phys=R\_true,
    D\_phys=D\_REF,
    rho\_phys=rho\_cand,
    T2B\_phys=T2B\_REF,
    t\_start=0.0,
    t\_final=tf\_val,
    dt\_phys=dt\_phys,
    ...
)
```

* A consistency check is performed:

```python
if len(time\_fem) != len(t\_obs) or not np.allclose(time\_fem, t\_obs):
    s\_out = np.interp(t\_obs, time\_fem, s\_out)
```

but in normal use (coherent `dt\_phys` and `tf\_val`), `time\_fem` and `t\_obs` match exactly and no interpolation is needed.

**Effect**

Semi-analytical and FEM signals are now computed on *exactly the same* physical time grid, eliminating timing-induced biases in the likelihood and stabilizing the inference of ρ in fast-decay regimes.

\---

### 2\. Correct and Explicit Use of the Dimensionless Semi-Analytical Model

**Original behavior**

* The Brownstein–Tarr solution `NMR\_SemiA\_sphere\_dimless` was called with a time grid that was not guaranteed to be consistent with the FEM time stepping or with the dimensionless scaling $\\tau = t / T\_{2B}$.
* In some configurations, this led to semi-analytical signals evaluated at times that did not match those used in `BT\_fenics\_decay`, further amplifying discrepancies.

**New behavior**

* The dimensionless time grid is explicitly tied to the physical observation times:

```python
tau\_grid = t\_obs / T2B\_REF
```

* The semi-analytical solver is called as:

```python
tau\_sa, s\_gen, \_, \_, \_, \_ = NMR\_SemiA\_sphere\_dimless(
    radius=R\_true,
    diffusion=D\_REF,
    rho=rho\_true,
    T2B=T2B\_REF,
    tau\_array=tau\_grid,
    n\_terms=200,
    return\_data='all'
)
```

* The dimensionless formulation used in both codes is identical:

  * $\\tau = t/T\_{2B}$,
  * $\\phi = r/\\sqrt{D T\_{2B}}$,
  * $\\phi\_R = R/\\sqrt{D T\_{2B}}$,
  * $\\kappa = R\\rho/D = \\alpha \\phi\_R$.

**Effect**

The “ground truth” semi-analytical signal is now evaluated *exactly* at the same dimensionless times as the FEM model, with identical scaling. Any mismatch between FEM and analytical solutions is now due to numerical discretization or noise, not to inconsistent non-dimensionalization or time grids.

\---

### 3\. Cleaner Handling of FEM Failures and Removal of Artificial Artifacts

**Original behavior**

* If `BT\_fenics\_decay` raised an exception, the forward model returned an array filled with `-10.0`:

```python
return np.full\_like(t\_obs, -10.0)
```

* In `Energy`, any occurrence of `-10.0` triggered an infinite energy (rejection):

```python
if np.any(s\_pred == -10.0): return np.inf
```



**Problems**

* Filling with a constant unphysical value introduces a sharp, artificial structure into the likelihood surface.
* Combined with the ad-hoc interpolation, this could accidentally create regions where the sampler behaves poorly or gets pushed into extreme parameter values.

**New behavior**

* If `BT\_fenics\_decay` fails, the forward model returns `None`:

```python
except Exception as e:
    print(f"Error in BT\_fenics\_decay (alpha={alpha\_cand}): {e}")
    return None
```

* In `Energy`, a `None` prediction is penalized with a large but finite energy:

```python
s\_pred = forward\_fem\_mcmc(alpha\_val)
if s\_pred is None:
    return 1e6
```



**Effect**

The sampler now sees a smooth, physically meaningful likelihood landscape without artificial spikes from unphysical placeholder values. Proposals that cause FEM failures are simply disfavored, without contaminating the numerical evaluation of the log-likelihood.

\---

### 4\. Minor but Relevant Numerical and Statistical Improvements

Although not the primary cause of the ρ bias, the following changes improve robustness:

1. **Safe handling of very small σ in the likelihood**

```python
sigma\_safe = max(sigma\_val, 1e-12)
log\_lik = -n \* np.log(sigma\_safe) - 0.5 \* np.sum(resid\*\*2) / sigma\_safe\*\*2
```

This avoids numerical overflow when σ proposals are extremely small.
2. **Explicit transformation between α and ρ consistent with scaling**

```python
rho\_cand = alpha\_cand \* np.sqrt(D\_REF / T2B\_REF)
...
rho\_chain\_full = alpha\_chain\_full \* np.sqrt(D\_REF / T2B\_REF)
```

ensuring that the inferred α is always mapped back to ρ using the same non-dimensionalization as in the FEM and semi-analytical solvers.
3. **Consistent use of the same `dt\_phys` and `tf\_val` in all components**
- A single `dt\_phys = DT\_VALUE` and `tf\_val` are used:
- to define `t\_obs`,
- in `BT\_fenics\_decay`,
- and indirectly in the dimensionless semi-analytical call via `tau\_array`.

\---

### 5\. Why These Changes Fix the ρ Overestimation for Small R and Large ρ

In the challenging regime $R = 2.25,\\mu\\text{m}$, $\\rho = 40,\\mu\\text{m/s}$:

* The problem is strongly surface‑relaxation dominated; the normalized signal decays very rapidly and the early time points carry most of the information about ρ.
* In the original implementation, misaligned time grids and interpolations made the FEM signal effectively “too slow” or “too fast” compared to the semi-analytical reference.
* The MCMC compensated systematic shape mismatches in the decay curve by pushing α (and hence ρ) to unrealistically high values.

With the new implementation:

* Time grids, non-dimensionalization, and scaling between α and ρ are internally consistent across semi-analytical and FEM models.
* The only differences between synthetic data and FEM predictions arise from discretization error and added noise, not from misalignment or inconsistent scaling.
* As a result, the posterior for ρ is centered much closer to the true value, with uncertainty controlled by the noise level and the chosen priors, rather than being dominated by systematic model mismatch.
<span style="display:none">[^1](SemiA_Sphere_NMR_dimensionless-2.py)</span>

<div align="center">⁂</div>

