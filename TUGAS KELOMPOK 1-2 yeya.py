import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# Parameter
N = 1500
dt = 0.005
steps = 200
h = 0.08
mass = 1.0 / N
G = 1.0
k = 0.5

# Inisialisasi
theta = np.random.rand(N) * 2*np.pi
r = np.sqrt(np.random.rand(N)) * 0.5
pos = np.vstack((r*np.cos(theta), r*np.sin(theta))).T + 0.5
vel = np.zeros((N,2))

# --- PARALLEL FUNCTIONS ---

@njit(parallel=True)
def compute_density(pos, rho, N, mass, h):
    for i in prange(N):
        total = 0.0
        for j in range(N):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            r = np.sqrt(dx*dx + dy*dy)
            total += mass * np.exp(-(r*r)/(h*h))
        rho[i] = total


@njit(parallel=True)
def compute_pressure(rho, pressure, N, k):
    for i in prange(N):
        pressure[i] = k * (rho[i] - 1.0)


@njit(parallel=True)
def compute_forces(pos, vel, rho, pressure, forces, N, mass, h, G):
    for i in prange(N):
        fx, fy = 0.0, 0.0
        for j in range(N):
            if i != j:
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-5

                # Pressure
                if dist < h:
                    grad = -2 * np.exp(-(dist**2)/(h**2)) / (h**2)
                    fx += -mass * (pressure[i]+pressure[j])/(2*rho[j]) * grad * dx
                    fy += -mass * (pressure[i]+pressure[j])/(2*rho[j]) * grad * dy

                # Gravity
                fx += -G * mass * dx / (dist**3)
                fy += -G * mass * dy / (dist**3)

        forces[i,0] = fx
        forces[i,1] = fy


# Storage
rho = np.zeros(N)
pressure = np.zeros(N)
forces = np.zeros((N,2))

# Simulasi
plt.ion()
for step in range(steps):
    compute_density(pos, rho, N, mass, h)
    compute_pressure(rho, pressure, N, k)
    compute_forces(pos, vel, rho, pressure, forces, N, mass, h, G)

    vel += dt * forces
    pos += dt * vel

    plt.clf()
    plt.scatter(pos[:,0], pos[:,1], s=5)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title(f"Parallel Collapse Step {step}")
    plt.pause(0.01)

plt.ioff()
plt.show()
