"""
Visualization and Comparison: PINN vs Traditional Methods for Schrödinger Equation

Generates:
1. Wavefunction evolution comparison (PINN vs Crank-Nicolson vs Split-Step Fourier)
2. Probability density heatmaps
3. Error analysis
4. Animated visualization
5. Tunneling probability over time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import os
import time

plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_results():
    """Load all saved results."""
    results = {}
    try:
        data = np.load('results/pinn_results.npz')
        results['pinn'] = {
            'x': data['x'], 't': data['t'],
            'psi_r': data['psi_r'], 'psi_i': data['psi_i'],
            'prob': data['prob_density']
        }
        print(f"Loaded PINN results: x={results['pinn']['x'].shape}, t={results['pinn']['t'].shape}")
    except FileNotFoundError:
        print("PINN results not found, will skip PINN plots")

    try:
        data = np.load('results/cn_results.npz')
        results['cn'] = {
            'x': data['x'], 't': data['t'],
            'psi_real': data['psi_real'], 'psi_imag': data['psi_imag'],
            'prob': data['prob_density']
        }
        print(f"Loaded CN results: x={results['cn']['x'].shape}, t={results['cn']['t'].shape}")
    except FileNotFoundError:
        pass

    try:
        data = np.load('results/ssf_results.npz')
        results['ssf'] = {
            'x': data['x'], 't': data['t'],
            'psi_real': data['psi_real'], 'psi_imag': data['psi_imag'],
            'prob': data['prob_density']
        }
        print(f"Loaded SSF results: x={results['ssf']['x'].shape}, t={results['ssf']['t'].shape}")
    except FileNotFoundError:
        pass

    return results


def potential_barrier(x, t):
    V0 = 5.0
    eps = 0.3
    omega = 3.0
    return V0 * (1.0 / np.cosh(x))**2 * (1.0 + eps * np.sin(omega * t))


def figure1_wavefunction_snapshots(results):
    """Figure 1: Wavefunction snapshots at multiple time points."""
    print("Generating Figure 1: Wavefunction snapshots...")

    if 'pinn' not in results or 'ssf' not in results:
        print("  Missing data, skipping")
        return

    times = [0, 1.25, 2.5, 3.75, 5.0]
    n_times = len(times)
    fig, axes = plt.subplots(n_times, 1, figsize=(12, 14))

    for i, t_target in enumerate(times):
        ax = axes[i]

        # PINN results
        t_idx_pinn = np.argmin(np.abs(results['pinn']['t'] - t_target))
        x_pinn = results['pinn']['x']
        prob_pinn = results['pinn']['prob'][t_idx_pinn, :]
        ax.plot(x_pinn, prob_pinn, 'b-', linewidth=1.8, label='PINN')

        # SSF results (reference)
        t_idx_ssf = np.argmin(np.abs(results['ssf']['t'] - t_target))
        x_ssf = results['ssf']['x']
        prob_ssf = results['ssf']['prob'][t_idx_ssf, :]
        ax.plot(x_ssf, prob_ssf, 'r--', linewidth=1.5, label='SSF (Reference)')

        # Potential (scaled)
        V = potential_barrier(x_ssf, t_target)
        V_scaled = V / V.max() * prob_ssf.max() * 0.8
        ax.fill_between(x_ssf, 0, V_scaled, alpha=0.2, color='gray', label='V(x,t) (scaled)')

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('|ψ|²', fontsize=11)
        ax.set_title(f't = {t_target:.2f}', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(-10, 10)

    plt.suptitle('Wavefunction Evolution: PINN vs Traditional Method', fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig('results/fig1_wavefunction_snapshots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done")


def figure2_heatmap_comparison(results):
    """Figure 2: Spacetime heatmap of probability density."""
    print("Generating Figure 2: Heatmap comparison...")

    if 'pinn' not in results or 'ssf' not in results:
        print("  Missing data, skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, data) in zip(axes,
                                 [('PINN', results.get('pinn')),
                                  ('Crank-Nicolson', results.get('cn')),
                                  ('Split-Step Fourier', results.get('ssf'))]):
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(name)
            continue

        X, T = np.meshgrid(data['x'], data['t'])
        im = ax.pcolormesh(X, T, data['prob'], shading='auto', cmap='viridis')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('t', fontsize=11)
        ax.set_title(name, fontsize=12)
        plt.colorbar(im, ax=ax, label='|ψ|²')

    plt.suptitle('Probability Density |ψ(x,t)|²: Method Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fig2_heatmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done")


def figure3_error_analysis(results):
    """Figure 3: Error analysis between PINN and reference."""
    print("Generating Figure 3: Error analysis...")

    if 'pinn' not in results or 'ssf' not in results:
        print("  Missing data, skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Interpolate PINN to SSF grid for comparison
    from scipy.interpolate import RectBivariateSpline

    x_pinn, t_pinn = results['pinn']['x'], results['pinn']['t']
    x_ssf, t_ssf = results['ssf']['x'], results['ssf']['t']
    prob_pinn = results['pinn']['prob']
    prob_ssf = results['ssf']['prob']

    # Interpolate PINN onto the SSF grid
    interp = RectBivariateSpline(t_pinn, x_pinn, prob_pinn)
    prob_pinn_interp = interp(t_ssf, x_ssf)

    # Error map
    error = np.abs(prob_pinn_interp - prob_ssf)
    X_ssf, T_ssf = np.meshgrid(x_ssf, t_ssf)

    im = axes[0, 0].pcolormesh(X_ssf, T_ssf, error, shading='auto', cmap='hot')
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('t', fontsize=11)
    axes[0, 0].set_title('Absolute Error |PINN - SSF|', fontsize=12)
    plt.colorbar(im, ax=axes[0, 0])

    # Relative error
    rel_error = error / (prob_ssf + 1e-10)
    rel_error = np.clip(rel_error, 0, 1)
    im = axes[0, 1].pcolormesh(X_ssf, T_ssf, rel_error, shading='auto', cmap='hot')
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('t', fontsize=11)
    axes[0, 1].set_title('Relative Error |PINN - SSF| / |SSF|', fontsize=12)
    plt.colorbar(im, ax=axes[0, 1])

    # Error vs time (L2 norm)
    l2_error = np.sqrt(np.trapz(error**2, x_ssf, axis=1))
    axes[1, 0].plot(t_ssf, l2_error, 'b-', linewidth=1.5)
    axes[1, 0].set_xlabel('t', fontsize=11)
    axes[1, 0].set_ylabel('L² Error', fontsize=11)
    axes[1, 0].set_title('L² Error vs Time', fontsize=12)
    axes[1, 0].set_yscale('log')

    # Norm preservation check
    norm_pinn = np.trapz(prob_pinn, x_pinn, axis=1)
    norm_ssf = np.trapz(prob_ssf, x_ssf, axis=1)
    axes[1, 1].plot(t_pinn, norm_pinn, 'b-', linewidth=1.5, label='PINN')
    axes[1, 1].plot(t_ssf, norm_ssf, 'r--', linewidth=1.5, label='SSF')
    axes[1, 1].axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('t', fontsize=11)
    axes[1, 1].set_ylabel('∫|ψ|² dx', fontsize=11)
    axes[1, 1].set_title('Norm Conservation Check', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_ylim(0.9, 1.1)

    plt.suptitle('Error Analysis: PINN vs Split-Step Fourier Reference', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fig3_error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done")


def figure4_tunneling_analysis(results):
    """Figure 4: Tunneling probability analysis."""
    print("Generating Figure 4: Tunneling analysis...")

    if 'ssf' not in results:
        print("  Missing data, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x_ssf = results['ssf']['x']
    t_ssf = results['ssf']['t']
    prob_ssf = results['ssf']['prob']

    # Transmission probability (probability on the right side of barrier)
    barrier_center = 0.0
    right_mask = x_ssf > barrier_center
    transm_prob = np.trapz(prob_ssf[:, right_mask], x_ssf[right_mask], axis=1)

    axes[0].plot(t_ssf, transm_prob, 'b-', linewidth=2)
    axes[0].set_xlabel('Time t', fontsize=12)
    axes[0].set_ylabel('Transmission Probability', fontsize=12)
    axes[0].set_title('Tunneling Probability vs Time', fontsize=13)
    axes[0].grid(True, alpha=0.3)

    # Reflection probability
    left_mask = x_ssf < barrier_center
    refl_prob = np.trapz(prob_ssf[:, left_mask], x_ssf[left_mask], axis=1)
    axes[0].plot(t_ssf, refl_prob, 'r--', linewidth=2, label='Reflection')
    axes[0].plot(t_ssf, transm_prob + refl_prob, 'k:', linewidth=1,
                 label='Total', alpha=0.7)
    axes[0].legend(fontsize=10)

    # Potential with time modulation
    t_dense = np.linspace(0, 5, 500)
    V_barrier_top = np.array([potential_barrier(np.array([0.0]), t)[0] for t in t_dense])

    axes[1].plot(t_dense, V_barrier_top, 'g-', linewidth=2)
    axes[1].set_xlabel('Time t', fontsize=12)
    axes[1].set_ylabel('V(0, t)', fontsize=12)
    axes[1].set_title('Barrier Height Modulation V(0, t) = V₀(1+εsin(ωt))', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Quantum Tunneling Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fig4_tunneling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done")


def figure5_loss_curve():
    """Figure 5: PINN training loss curve."""
    print("Generating Figure 5: Loss curve...")

    try:
        loss_arr = np.loadtxt('results/loss_history.csv', delimiter=',', skiprows=1)
    except (FileNotFoundError, OSError):
        print("  Loss history not found, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(len(loss_arr))
    ax.semilogy(epochs, loss_arr[:, 0], 'k-', linewidth=1.5, label='Total Loss')
    ax.semilogy(epochs, loss_arr[:, 1], 'b-', linewidth=1, alpha=0.7, label='PDE Loss')
    ax.semilogy(epochs, loss_arr[:, 2], 'r-', linewidth=1, alpha=0.7, label='IC Loss')
    ax.semilogy(epochs, loss_arr[:, 3], 'g-', linewidth=1, alpha=0.7, label='BC Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('PINN Training Convergence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig5_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Done")


def animate_evolution(results):
    """Create animation of wavefunction evolution."""
    print("Generating animation...")

    if 'pinn' not in results or 'ssf' not in results:
        print("  Missing data for animation, skipping")
        return

    x_pinn = results['pinn']['x']
    t_pinn = results['pinn']['t']
    prob_pinn = results['pinn']['prob']

    x_ssf = results['ssf']['x']
    t_ssf = results['ssf']['t']
    prob_ssf = results['ssf']['prob']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Interpolate PINN to match SSF time steps
    from scipy.interpolate import RectBivariateSpline
    interp = RectBivariateSpline(t_pinn, x_pinn, prob_pinn)

    n_frames = min(200, len(t_ssf))
    frame_indices = np.linspace(0, len(t_ssf) - 1, n_frames, dtype=int)

    def update(frame_idx):
        ax.clear()
        t_idx = frame_indices[frame_idx]
        t_val = t_ssf[t_idx]

        # SSF data
        ax.plot(x_ssf, prob_ssf[t_idx, :], 'r-', linewidth=1.8, label='SSF (Reference)')

        # PINN data
        t_pinn_idx = np.argmin(np.abs(t_pinn - t_val))
        ax.plot(x_pinn, prob_pinn[t_pinn_idx, :], 'b--', linewidth=1.8, label='PINN')

        # Potential (scaled)
        V = potential_barrier(x_ssf, t_val)
        if V.max() > 0:
            V_scaled = V / V.max() * max(prob_ssf[t_idx, :].max(), prob_pinn[t_pinn_idx, :].max()) * 0.7
            ax.fill_between(x_ssf, 0, V_scaled, alpha=0.2, color='gray', label='V(x,t)')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('|ψ|²', fontsize=12)
        ax.set_title(f'Wavefunction Evolution: t = {t_val:.2f}', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, None)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)
    anim.save('results/wavefunction_evolution.gif', writer='pillow', fps=10, dpi=100)
    plt.close()
    print("  Animation saved")


def main():
    """Main visualization routine."""
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("Visualization and Comparison Generator")
    print("=" * 60)

    results = load_results()

    if not results:
        print("No results loaded. Run solvers first.")
        return

    figure1_wavefunction_snapshots(results)
    figure2_heatmap_comparison(results)
    figure3_error_analysis(results)
    figure4_tunneling_analysis(results)
    figure5_loss_curve()
    animate_evolution(results)

    print("\nAll figures saved to 'results/' directory.")


if __name__ == '__main__':
    main()
