# Venturi Channel Flow: MPI + OpenMP-Style Python

This project simulates fluid flow through a Venturi channel using a D2Q9 lattice-Boltzmann method. The model is physically based: it evolves density and momentum according to the incompressible Navier-Stokes limit, uses no-slip bounce-back walls, a parabolic inlet, and a pressure outlet. The throat accelerates the flow and produces a lower-pressure region, matching the Venturi effect.

Parallelism:

- **MPI**: `mpi4py` splits the grid in the vertical direction and exchanges halo rows between ranks.
- **OpenMP-style threading**: Numba runs the heavy loops with `prange` and requests the `omp` threading layer.
- **Animation**: rank 0 gathers fields and saves an animated speed/pressure visualization.

## Install

Use Python 3.10 or newer. The solver needs Numba for OpenMP-style threaded kernels, so make sure your Python environment can install binary wheels.

Install an MPI implementation first:

- Windows: Microsoft MPI or MPICH/MSYS2.
- Linux: OpenMPI or MPICH.
- macOS: OpenMPI from Homebrew.

Then install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If Numba reports that the `omp` threading layer cannot be loaded, install an OpenMP runtime for your platform or remove `NUMBA_THREADING_LAYER=omp` from your environment to let Numba choose a fallback threading backend.

## Run

Serial run with threaded kernels:

```bash
python venturi_lbm_mpi_openmp.py --threads 4 --steps 2500 --output venturi.gif
```

MPI + OpenMP-style hybrid run:

```bash
mpiexec -n 4 python venturi_lbm_mpi_openmp.py --threads 2 --steps 3000 --output venturi_mpi.gif
```

Live preview on rank 0:

```bash
mpiexec -n 4 python venturi_lbm_mpi_openmp.py --threads 2 --show --steps 2000
```

Run faster without animation:

```bash
mpiexec -n 4 python venturi_lbm_mpi_openmp.py --threads 2 --no-animation --steps 5000
```

## Useful Parameters

- `--reynolds 120`: Reynolds number based on the Venturi throat height.
- `--u-max 0.055`: peak lattice inlet speed. Keep below `0.1` to satisfy the low-Mach assumption.
- `--constriction 0.55`: fractional channel half-height reduction at the throat.
- `--nx 360 --ny 140`: grid resolution.
- `--plot-every 25`: animation frame interval.

## Notes On Physical Validity

The lattice-Boltzmann method is a real fluid solver, not a purely visual particle effect. In the low-Mach, moderate-Reynolds regime it recovers the incompressible Navier-Stokes equations. For stable and physically meaningful results:

- Use `--u-max < 0.1`.
- Keep the relaxation time `tau` above `0.505`; the script checks this.
- Use moderate Reynolds numbers, for example `50` to `250`, unless you increase resolution.
- Interpret the default units as lattice units. Pressure is computed with the LBM equation of state `p = rho / 3`.
