#!/usr/bin/env python3
"""
MPI + OpenMP-style Venturi channel flow simulation in Python.

Numerics:
  - D2Q9 lattice-Boltzmann method with BGK collision.
  - No-slip walls through bounce-back.
  - Parabolic velocity inlet and pressure outlet.
  - MPI domain decomposition in the y direction.
  - Numba parallel loops request the OpenMP threading layer.

Run examples:
  python venturi_lbm_mpi_openmp.py --steps 1500 --output venturi.gif
  mpiexec -n 4 python venturi_lbm_mpi_openmp.py --threads 4 --steps 3000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")

import numpy as np

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - useful on machines without MPI installed
    MPI = None

try:
    from numba import get_num_threads, njit, prange, set_num_threads, threading_layer
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Numba is required for the OpenMP-style threaded kernels.\n"
        "Install dependencies with: python -m pip install -r requirements.txt"
    ) from exc


CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, -1, 0, 1, -1, -1, 1, 1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CS2 = 1.0 / 3.0


class SerialComm:
    """Tiny mpi4py-like communicator for non-MPI runs."""

    rank = 0
    size = 1

    def Barrier(self) -> None:
        return None

    def gather(self, value, root=0):
        return [value]


def get_comm():
    if MPI is None:
        return SerialComm()
    return MPI.COMM_WORLD


def split_rows(ny: int, size: int, rank: int) -> tuple[int, int]:
    base = ny // size
    extra = ny % size
    local_ny = base + (1 if rank < extra else 0)
    y0 = rank * base + min(rank, extra)
    return y0, local_ny


def make_venturi_solid(
    nx: int,
    ny: int,
    y0: int,
    local_ny: int,
    open_fraction: float,
    constriction: float,
    throat_center: float,
    throat_sigma: float,
) -> np.ndarray:
    """Return solid mask for local rows plus one halo row on each side."""

    local_with_halo = np.arange(y0 - 1, y0 + local_ny + 1, dtype=np.float64)[:, None]
    x = np.arange(nx, dtype=np.float64)[None, :]

    center_y = 0.5 * (ny - 1)
    base_half_height = 0.5 * ny * open_fraction
    x_center = throat_center * (nx - 1)
    sigma = throat_sigma * nx
    narrowing = constriction * np.exp(-0.5 * ((x - x_center) / sigma) ** 2)
    half_height = base_half_height * (1.0 - narrowing)

    solid = (local_with_halo < center_y - half_height) | (local_with_halo > center_y + half_height)
    solid |= local_with_halo < 0
    solid |= local_with_halo > ny - 1
    return np.ascontiguousarray(solid)


def make_inlet_profile(
    solid: np.ndarray,
    nx: int,
    ny: int,
    y0: int,
    local_ny: int,
    open_fraction: float,
    constriction: float,
    throat_center: float,
    throat_sigma: float,
    u_max: float,
) -> np.ndarray:
    """Parabolic inlet profile on x=0 for local rows, including halos."""

    global_y = np.arange(y0 - 1, y0 + local_ny + 1, dtype=np.float64)
    center_y = 0.5 * (ny - 1)
    base_half_height = 0.5 * ny * open_fraction
    x_center = throat_center * (nx - 1)
    sigma = throat_sigma * nx
    narrowing_at_inlet = constriction * np.exp(-0.5 * ((0.0 - x_center) / sigma) ** 2)
    inlet_half_height = base_half_height * (1.0 - narrowing_at_inlet)
    lo = center_y - inlet_half_height
    hi = center_y + inlet_half_height

    eta = (global_y - lo) / max(hi - lo, 1.0)
    fluid = (~solid[:, 0]) & (eta >= 0.0) & (eta <= 1.0)
    profile = np.zeros(solid.shape[0], dtype=np.float64)
    profile[fluid] = 4.0 * u_max * eta[fluid] * (1.0 - eta[fluid])
    return np.ascontiguousarray(profile)


@njit(cache=True, fastmath=True)
def _feq(q: int, rho: float, ux: float, uy: float) -> float:
    cu = 3.0 * (CX[q] * ux + CY[q] * uy)
    uu = ux * ux + uy * uy
    return W[q] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * uu)


@njit(parallel=True, cache=True, fastmath=True)
def initialize_distribution(f: np.ndarray, solid: np.ndarray, inlet_profile: np.ndarray) -> None:
    ny_local = f.shape[0] - 2
    nx = f.shape[1]
    for i in prange(1, ny_local + 1):
        ux0 = inlet_profile[i] * 0.25
        for j in range(nx):
            ux = ux0 if not solid[i, j] else 0.0
            uy = 0.0
            rho = 1.0
            for q in range(9):
                f[i, j, q] = _feq(q, rho, ux, uy)


@njit(parallel=True, cache=True, fastmath=True)
def collide(f: np.ndarray, f_post: np.ndarray, solid: np.ndarray, omega: float) -> None:
    ny_local = f.shape[0] - 2
    nx = f.shape[1]
    for i in prange(1, ny_local + 1):
        for j in range(nx):
            if solid[i, j]:
                for q in range(9):
                    f_post[i, j, q] = f[i, j, q]
                continue

            rho = 0.0
            ux = 0.0
            uy = 0.0
            for q in range(9):
                fq = f[i, j, q]
                rho += fq
                ux += fq * CX[q]
                uy += fq * CY[q]
            ux /= rho
            uy /= rho

            for q in range(9):
                f_post[i, j, q] = f[i, j, q] - omega * (f[i, j, q] - _feq(q, rho, ux, uy))


@njit(parallel=True, cache=True, fastmath=True)
def stream_pull_bounce(f_post: np.ndarray, f_next: np.ndarray, solid: np.ndarray) -> None:
    ny_local = f_post.shape[0] - 2
    nx = f_post.shape[1]

    for i in prange(1, ny_local + 1):
        for j in range(nx):
            if solid[i, j]:
                for q in range(9):
                    f_next[i, j, q] = _feq(q, 1.0, 0.0, 0.0)
                continue

            for q in range(9):
                si = i - CY[q]
                sj = j - CX[q]

                if sj < 0 or sj >= nx:
                    f_next[i, j, q] = f_post[i, j, OPP[q]]
                elif solid[si, sj]:
                    f_next[i, j, q] = f_post[i, j, OPP[q]]
                else:
                    f_next[i, j, q] = f_post[si, sj, q]


@njit(parallel=True, cache=True, fastmath=True)
def apply_zou_he_boundaries(
    f: np.ndarray,
    solid: np.ndarray,
    inlet_profile: np.ndarray,
    rho_out: float,
) -> None:
    ny_local = f.shape[0] - 2
    nx = f.shape[1]
    x_left = 0
    x_right = nx - 1

    for i in prange(1, ny_local + 1):
        if not solid[i, x_left]:
            ux = inlet_profile[i]
            uy = 0.0
            rho = (
                f[i, x_left, 0]
                + f[i, x_left, 2]
                + f[i, x_left, 4]
                + 2.0 * (f[i, x_left, 3] + f[i, x_left, 6] + f[i, x_left, 7])
            ) / (1.0 - ux)

            f[i, x_left, 1] = f[i, x_left, 3] + (2.0 / 3.0) * rho * ux
            f[i, x_left, 5] = (
                f[i, x_left, 7]
                + 0.5 * (f[i, x_left, 4] - f[i, x_left, 2])
                + (1.0 / 6.0) * rho * ux
                + 0.5 * rho * uy
            )
            f[i, x_left, 8] = (
                f[i, x_left, 6]
                + 0.5 * (f[i, x_left, 2] - f[i, x_left, 4])
                + (1.0 / 6.0) * rho * ux
                - 0.5 * rho * uy
            )

        if not solid[i, x_right]:
            rho = rho_out
            uy = 0.0
            ux = -1.0 + (
                f[i, x_right, 0]
                + f[i, x_right, 2]
                + f[i, x_right, 4]
                + 2.0 * (f[i, x_right, 1] + f[i, x_right, 5] + f[i, x_right, 8])
            ) / rho

            f[i, x_right, 3] = f[i, x_right, 1] - (2.0 / 3.0) * rho * ux
            f[i, x_right, 7] = (
                f[i, x_right, 5]
                + 0.5 * (f[i, x_right, 2] - f[i, x_right, 4])
                - (1.0 / 6.0) * rho * ux
                - 0.5 * rho * uy
            )
            f[i, x_right, 6] = (
                f[i, x_right, 8]
                + 0.5 * (f[i, x_right, 4] - f[i, x_right, 2])
                - (1.0 / 6.0) * rho * ux
                + 0.5 * rho * uy
            )


@njit(parallel=True, cache=True, fastmath=True)
def macroscopic(
    f: np.ndarray,
    solid: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    speed: np.ndarray,
    pressure: np.ndarray,
) -> None:
    ny_local = f.shape[0] - 2
    nx = f.shape[1]

    for i_local in prange(ny_local):
        i = i_local + 1
        for j in range(nx):
            if solid[i, j]:
                rho[i_local, j] = np.nan
                ux[i_local, j] = 0.0
                uy[i_local, j] = 0.0
                speed[i_local, j] = np.nan
                pressure[i_local, j] = np.nan
                continue

            r = 0.0
            u = 0.0
            v = 0.0
            for q in range(9):
                fq = f[i, j, q]
                r += fq
                u += fq * CX[q]
                v += fq * CY[q]

            u /= r
            v /= r
            rho[i_local, j] = r
            ux[i_local, j] = u
            uy[i_local, j] = v
            speed[i_local, j] = np.sqrt(u * u + v * v)
            pressure[i_local, j] = CS2 * r


def exchange_halos(f_post: np.ndarray, comm, rank: int, size: int) -> None:
    if size == 1:
        return
    top = rank - 1 if rank > 0 else MPI.PROC_NULL
    bottom = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    comm.Sendrecv(
        sendbuf=np.ascontiguousarray(f_post[1, :, :]),
        dest=top,
        sendtag=10,
        recvbuf=f_post[-1, :, :],
        source=bottom,
        recvtag=10,
    )
    comm.Sendrecv(
        sendbuf=np.ascontiguousarray(f_post[-2, :, :]),
        dest=bottom,
        sendtag=20,
        recvbuf=f_post[0, :, :],
        source=top,
        recvtag=20,
    )


def gather_field(local_field: np.ndarray, comm, rank: int, size: int) -> np.ndarray | None:
    pieces = comm.gather(np.ascontiguousarray(local_field), root=0)
    if rank != 0:
        return None
    return np.vstack(pieces)


def compute_reynolds_viscosity(
    ny: int,
    open_fraction: float,
    constriction: float,
    u_max: float,
    reynolds: float,
) -> tuple[float, float, float]:
    throat_height = max(4.0, ny * open_fraction * (1.0 - constriction))
    u_mean = (2.0 / 3.0) * u_max
    nu = u_mean * throat_height / reynolds
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau
    return nu, tau, omega


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPI + OpenMP-style D2Q9 LBM simulation of flow through a Venturi channel."
    )
    parser.add_argument("--nx", type=int, default=360, help="Lattice cells in the streamwise direction.")
    parser.add_argument("--ny", type=int, default=140, help="Lattice cells in the cross-channel direction.")
    parser.add_argument("--steps", type=int, default=2500, help="Number of time steps.")
    parser.add_argument("--plot-every", type=int, default=25, help="Animation frame interval.")
    parser.add_argument("--threads", type=int, default=0, help="OpenMP threads per MPI rank through Numba.")
    parser.add_argument("--u-max", type=float, default=0.055, help="Peak inlet speed in lattice units. Keep < 0.1.")
    parser.add_argument("--reynolds", type=float, default=120.0, help="Reynolds number based on throat height.")
    parser.add_argument("--rho-out", type=float, default=1.0, help="Outlet density, giving p = rho / 3.")
    parser.add_argument("--open-fraction", type=float, default=0.78, help="Inlet/outlet opening as a fraction of ny.")
    parser.add_argument("--constriction", type=float, default=0.55, help="Fractional reduction of half-height at throat.")
    parser.add_argument("--throat-center", type=float, default=0.52, help="Throat center as a fraction of nx.")
    parser.add_argument("--throat-sigma", type=float, default=0.16, help="Venturi narrowing width as a fraction of nx.")
    parser.add_argument("--output", type=Path, default=Path("venturi.gif"), help="GIF/MP4 animation path.")
    parser.add_argument("--no-animation", action="store_true", help="Run without saving an animation.")
    parser.add_argument("--show", action="store_true", help="Show a live matplotlib window on rank 0.")
    parser.add_argument("--dpi", type=int, default=120, help="Animation DPI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    comm = get_comm()
    rank = comm.rank
    size = comm.size

    if args.threads > 0:
        set_num_threads(args.threads)

    if args.u_max >= 0.1 and rank == 0:
        print("Warning: --u-max >= 0.1 can violate the low-Mach assumption used by LBM.", file=sys.stderr)

    y0, local_ny = split_rows(args.ny, size, rank)
    if local_ny < 1:
        raise SystemExit(f"MPI rank {rank} received no rows. Use <= {args.ny} ranks.")

    solid = make_venturi_solid(
        args.nx,
        args.ny,
        y0,
        local_ny,
        args.open_fraction,
        args.constriction,
        args.throat_center,
        args.throat_sigma,
    )
    inlet_profile = make_inlet_profile(
        solid,
        args.nx,
        args.ny,
        y0,
        local_ny,
        args.open_fraction,
        args.constriction,
        args.throat_center,
        args.throat_sigma,
        args.u_max,
    )
    nu, tau, omega = compute_reynolds_viscosity(
        args.ny, args.open_fraction, args.constriction, args.u_max, args.reynolds
    )

    if tau <= 0.505:
        raise SystemExit(
            f"Unstable relaxation time tau={tau:.4f}. Lower --reynolds or increase --u-max/--ny."
        )

    f = np.zeros((local_ny + 2, args.nx, 9), dtype=np.float64)
    f_post = np.zeros_like(f)
    f_next = np.zeros_like(f)
    rho = np.zeros((local_ny, args.nx), dtype=np.float64)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    speed = np.zeros_like(rho)
    pressure = np.zeros_like(rho)

    initialize_distribution(f, solid, inlet_profile)
    comm.Barrier()

    writer = None
    fig = None
    image = None
    pressure_contour = None
    animation_enabled = not args.no_animation
    visual_enabled = animation_enabled or args.show

    if rank == 0:
        print(
            "Venturi LBM setup\n"
            f"  grid: {args.nx} x {args.ny}\n"
            f"  MPI ranks: {size}\n"
            f"  Numba threading layer: {threading_layer() if get_num_threads() else 'unknown'}\n"
            f"  threads per rank: {get_num_threads()}\n"
            f"  Reynolds number: {args.reynolds:.2f}\n"
            f"  lattice viscosity: {nu:.6f}\n"
            f"  tau: {tau:.6f}, omega: {omega:.6f}\n"
            f"  output: {args.output if animation_enabled else 'disabled'}"
        )

    if visual_enabled:
        if rank == 0:
            import matplotlib.pyplot as plt
            from matplotlib import animation

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
            ax.set_title("Venturi channel speed field, pressure contours")
            ax.set_xlabel("x lattice cells")
            ax.set_ylabel("y lattice cells")
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#101010")
            image = ax.imshow(
                np.zeros((args.ny, args.nx)),
                origin="upper",
                cmap=cmap,
                interpolation="bilinear",
                vmin=0.0,
                vmax=max(args.u_max * 1.9, 0.01),
            )
            cbar = fig.colorbar(image, ax=ax, shrink=0.88)
            cbar.set_label("|u| lattice units")

            global_solid = make_venturi_solid(
                args.nx,
                args.ny,
                0,
                args.ny,
                args.open_fraction,
                args.constriction,
                args.throat_center,
                args.throat_sigma,
            )[1:-1]
            ax.contour(global_solid.astype(float), levels=[0.5], colors="white", linewidths=1.0)

            if animation_enabled:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                if args.output.suffix.lower() == ".mp4":
                    writer = animation.FFMpegWriter(fps=24, bitrate=1800)
                else:
                    writer = animation.PillowWriter(fps=24)
                writer.setup(fig, str(args.output), dpi=args.dpi)

    start = time.perf_counter()
    frame_count = 0

    try:
        for step in range(1, args.steps + 1):
            collide(f, f_post, solid, omega)
            exchange_halos(f_post, comm, rank, size)
            stream_pull_bounce(f_post, f_next, solid)
            apply_zou_he_boundaries(f_next, solid, inlet_profile, args.rho_out)
            f, f_next = f_next, f

            should_sample = step % args.plot_every == 0 or step == args.steps
            if visual_enabled and should_sample:
                macroscopic(f, solid, rho, ux, uy, speed, pressure)
                global_speed = gather_field(speed, comm, rank, size)
                global_pressure = gather_field(pressure, comm, rank, size)

                if rank == 0:
                    frame_count += 1
                    finite_speed = global_speed[np.isfinite(global_speed)]
                    finite_pressure = global_pressure[np.isfinite(global_pressure)]
                    if finite_speed.size:
                        vmax = max(float(np.percentile(finite_speed, 99.5)), args.u_max * 1.1)
                        image.set_clim(0.0, vmax)
                    image.set_data(np.ma.masked_invalid(global_speed))

                    if pressure_contour is not None:
                        try:
                            pressure_contour.remove()
                        except AttributeError:
                            for collection in pressure_contour.collections:
                                collection.remove()
                    if finite_pressure.size:
                        p_lo = float(np.percentile(finite_pressure, 5))
                        p_hi = float(np.percentile(finite_pressure, 95))
                        if p_hi > p_lo + 1e-10:
                            levels = np.linspace(p_lo, p_hi, 9)
                            pressure_contour = image.axes.contour(
                                global_pressure,
                                levels=levels,
                                colors="black",
                                linewidths=0.35,
                                alpha=0.45,
                            )
                        else:
                            pressure_contour = None

                    image.axes.set_title(f"Venturi channel flow, step {step}/{args.steps}")
                    if writer is not None:
                        writer.grab_frame()
                    if args.show:
                        import matplotlib.pyplot as plt

                        plt.pause(0.001)

                    if step % max(args.plot_every * 10, 1) == 0 or step == args.steps:
                        elapsed = time.perf_counter() - start
                        mlups = args.nx * args.ny * step / max(elapsed, 1e-12) / 1e6
                        print(f"step {step:6d} | {mlups:7.2f} MLUPS | frames {frame_count}")
            elif rank == 0 and should_sample and (
                step % max(args.plot_every * 10, 1) == 0 or step == args.steps
            ):
                elapsed = time.perf_counter() - start
                mlups = args.nx * args.ny * step / max(elapsed, 1e-12) / 1e6
                print(f"step {step:6d} | {mlups:7.2f} MLUPS")

    finally:
        if rank == 0 and writer is not None:
            writer.finish()
        if rank == 0 and args.show:
            import matplotlib.pyplot as plt

            plt.show()

    elapsed = time.perf_counter() - start
    if rank == 0:
        print(f"Finished in {elapsed:.2f} s.")
        if animation_enabled:
            print(f"Saved animation: {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
