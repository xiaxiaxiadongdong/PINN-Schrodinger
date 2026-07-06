"""
Traditional Numerical Solvers for the Time-Dependent Schrödinger Equation

Methods:
1. Crank-Nicolson Finite Difference Method
2. Split-Step Fourier Method (for comparison)

Solves: i ∂ψ/∂t = -∂²ψ/∂x² + V(x,t)ψ

The Crank-Nicolson method is unconditionally stable and second-order accurate.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time


class CrankNicolsonSolver:
    """Crank-Nicolson finite difference solver for the Schrödinger equation."""

    def __init__(self, x_min, x_max, nx, t_min, t_max, nt, V_func):
        """
        Args:
            x_min, x_max: spatial domain
            nx: number of spatial grid points
            t_min, t_max: temporal domain
            nt: number of time steps
            V_func: potential function V(x, t) -> array
        """
        self.x_min, self.x_max = x_min, x_max
        self.nx = nx
        self.t_min, self.t_max = t_min, t_max
        self.nt = nt
        self.V_func = V_func

        # Spatial grid
        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)

        # Temporal grid
        self.dt = (t_max - t_min) / nt
        self.t = np.linspace(t_min, t_max, nt + 1)

        # Initialize wavefunction storage
        self.psi = np.zeros((nt + 1, nx), dtype=complex)

    def initial_gaussian_wavepacket(self, sigma=1.0, x0=-5.0, k0=2.0):
        """Gaussian wave packet initial condition."""
        amp = 1.0 / ((2 * np.pi * sigma**2)**0.25)
        psi0 = amp * np.exp(-(self.x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * self.x)
        return psi0

    def build_matrices(self, n):
        """Build the Crank-Nicolson matrices at time step n.

        CN scheme: (I + i*dt/2*H)ψ^(n+1) = (I - i*dt/2*H)ψ^n
        where H = -d²/dx² + V
        """
        dx2 = self.dx**2
        dt = self.dt

        # Kinetic energy operator T = -d²/dx² (3-point stencil)
        main_diag = 2.0 / dx2 * np.ones(self.nx)
        off_diag = -1.0 / dx2 * np.ones(self.nx - 1)

        # H = T + V at time step n (for left-hand side)
        V_n = self.V_func(self.x, self.t[n])

        # Left matrix: A = I + i*dt/2*(T + V_n)
        A_main = 1.0 + 1j * dt / 2.0 * (main_diag + V_n)
        A_off = 1j * dt / 2.0 * off_diag

        # Right matrix: B = I - i*dt/2*(T + V_n)
        B_main = 1.0 - 1j * dt / 2.0 * (main_diag + V_n)
        B_off = -1j * dt / 2.0 * off_diag

        # Build sparse matrices
        A = sparse.diags([A_main, A_off, A_off], [0, 1, -1], format='csc')
        B = sparse.diags([B_main, B_off, B_off], [0, 1, -1], format='csr')

        return A, B

    def solve(self):
        """Solve the Schrödinger equation using Crank-Nicolson."""
        print("Solving with Crank-Nicolson method...")
        print(f"  Grid: {self.nx} spatial points × {self.nt} time steps")
        print(f"  dx = {self.dx:.4f}, dt = {self.dt:.4f}")

        start_time = time.time()

        # Initial condition
        self.psi[0, :] = self.initial_gaussian_wavepacket()

        for n in range(self.nt):
            if n % 500 == 0:
                print(f"  Step {n}/{self.nt} ({100*n/self.nt:.1f}%)")

            A, B = self.build_matrices(n)
            rhs = B.dot(self.psi[n, :])
            self.psi[n + 1, :] = spsolve(A, rhs)

        elapsed = time.time() - start_time
        print(f"Crank-Nicolson completed in {elapsed:.1f}s")

        # Compute probability density
        self.prob_density = np.abs(self.psi)**2

        return self.x, self.t, self.psi


class SplitStepFourierSolver:
    """Split-Step Fourier (SSF) method for the Schrödinger equation.

    More accurate than Crank-Nicolson for wave-like problems.
    Uses: ψ^(n+1) = exp(-i*dt/2*V) * F^{-1}[exp(-i*dt*k²) * F[exp(-i*dt/2*V) * ψ^n]]
    """

    def __init__(self, x_min, x_max, nx, t_min, t_max, nt, V_func):
        self.x_min, x_max = x_min, x_max
        self.nx = nx
        self.t_min, self.t_max = t_min, t_max
        self.nt = nt
        self.V_func = V_func

        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)
        self.dt = (t_max - t_min) / nt
        self.t = np.linspace(t_min, t_max, nt + 1)

        # Wavenumbers for Fourier space
        dk = 2 * np.pi / (x_max - x_min)
        self.k = np.fft.fftfreq(nx, self.dx) * 2 * np.pi
        self.k2 = self.k**2

        self.psi = np.zeros((nt + 1, nx), dtype=complex)

    def solve(self):
        """Solve using Split-Step Fourier method."""
        print("Solving with Split-Step Fourier method...")
        print(f"  Grid: {self.nx} spatial points × {self.nt} time steps")

        start_time = time.time()

        # Initial condition
        sigma, x0, k0 = 1.0, -5.0, 2.0
        amp = 1.0 / ((2 * np.pi * sigma**2)**0.25)
        self.psi[0, :] = amp * np.exp(-(self.x - x0)**2 / (4 * sigma**2)) * \
                         np.exp(1j * k0 * self.x)

        for n in range(self.nt):
            if n % 500 == 0:
                print(f"  Step {n}/{self.nt} ({100*n/self.nt:.1f}%)")

            # Half-step in position space
            V_half = self.V_func(self.x, self.t[n] + self.dt / 2)
            psi_temp = np.exp(-1j * self.dt / 2 * V_half) * self.psi[n, :]

            # Full step in momentum space
            psi_k = np.fft.fft(psi_temp)
            psi_k = np.exp(-1j * self.dt * self.k2) * psi_k
            psi_temp = np.fft.ifft(psi_k)

            # Half-step in position space
            V_half = self.V_func(self.x, self.t[n + 1] + self.dt / 2)
            self.psi[n + 1, :] = np.exp(-1j * self.dt / 2 * V_half) * psi_temp

        elapsed = time.time() - start_time
        print(f"Split-Step Fourier completed in {elapsed:.1f}s")

        self.prob_density = np.abs(self.psi)**2
        return self.x, self.t, self.psi


def potential_barrier(x, t):
    """Time-periodic perturbation potential for quantum tunneling.

    V(x,t) = V₀·sech²(x) · (1 + ε·sin(ωt))
    """
    V0 = 5.0
    eps = 0.3
    omega = 3.0
    return V0 * (1.0 / np.cosh(x))**2 * (1.0 + eps * np.sin(omega * t))


def main():
    """Run traditional solvers."""
    x_min, x_max = -10.0, 10.0
    nx = 512  # Higher resolution for reference solution
    t_min, t_max = 0.0, 5.0
    nt = 2000  # More time steps for accurate comparison

    os.makedirs('results', exist_ok=True)

    # Crank-Nicolson
    cn_solver = CrankNicolsonSolver(x_min, x_max, nx, t_min, t_max, nt,
                                     potential_barrier)
    x_cn, t_cn, psi_cn = cn_solver.solve()

    # Split-Step Fourier (more accurate reference)
    ssf_solver = SplitStepFourierSolver(x_min, x_max, nx, t_min, t_max, nt,
                                         potential_barrier)
    x_ssf, t_ssf, psi_ssf = ssf_solver.solve()

    prob_cn = np.abs(psi_cn)**2
    prob_ssf = np.abs(psi_ssf)**2

    # Save results
    np.savez('results/cn_results.npz',
             x=x_cn, t=t_cn, psi_real=psi_cn.real, psi_imag=psi_cn.imag,
             prob_density=prob_cn)
    np.savez('results/ssf_results.npz',
             x=x_ssf, t=t_ssf, psi_real=psi_ssf.real, psi_imag=psi_ssf.imag,
             prob_density=prob_ssf)

    # Quick comparison plot between CN and SSF at final time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Initial state
    axes[0, 0].plot(x_cn, prob_cn[0, :], 'b-', linewidth=1.5, label='t=0')
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('|ψ|²', fontsize=12)
    axes[0, 0].set_title('Initial Probability Density', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Mid-time
    mid = nt // 2
    axes[0, 1].plot(x_cn, prob_cn[mid, :], 'b-', linewidth=1.5, label=f'CN (t={t_cn[mid]:.1f})')
    axes[0, 1].plot(x_ssf, prob_ssf[mid, :], 'r--', linewidth=1.5, label=f'SSF (t={t_ssf[mid]:.1f})')
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('|ψ|²', fontsize=12)
    axes[0, 1].set_title(f'Probability Density at t = {t_cn[mid]:.1f}', fontsize=13)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Final time
    axes[1, 0].plot(x_cn, prob_cn[-1, :], 'b-', linewidth=1.5, label=f'CN (t={t_cn[-1]:.1f})')
    axes[1, 0].plot(x_ssf, prob_ssf[-1, :], 'r--', linewidth=1.5, label=f'SSF (t={t_ssf[-1]:.1f})')
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('|ψ|²', fontsize=12)
    axes[1, 0].set_title(f'Probability Density at t = {t_cn[-1]:.1f}', fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Potential
    V_plot = potential_barrier(x_cn, 0)
    axes[1, 1].plot(x_cn, V_plot, 'g-', linewidth=2, label='V(x,0)')
    V_plot_t = potential_barrier(x_cn, t_max)
    axes[1, 1].plot(x_cn, V_plot_t, 'orange', linewidth=2, linestyle='--', label=f'V(x,{t_max})')
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('V(x,t)', fontsize=12)
    axes[1, 1].set_title('Potential Function', fontsize=13)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Traditional Numerical Solutions of Schrödinger Equation', fontsize=15)
    plt.tight_layout()
    plt.savefig('results/traditional_solution.png', dpi=150)
    plt.close()

    print("Traditional solver results saved to 'results/' directory.")

    return x_cn, t_cn, psi_cn, prob_cn


if __name__ == '__main__':
    main()
