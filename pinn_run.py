"""
PINN for Time-Dependent Schrodinger Equation - Minimal Fast Version
Solves quantum tunneling with a periodic potential barrier.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        layer_dict = OrderedDict()
        for i in range(len(layers) - 1):
            layer_dict[f'linear_{i}'] = nn.Linear(layers[i], layers[i+1])
            if i < len(layers) - 2:
                layer_dict[f'act_{i}'] = self.activation
        self.network = nn.Sequential(layer_dict)
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=1))


# --- Physics ---
def potential_torch(x, t):
    V0, eps, omega = 5.0, 0.3, 3.0
    return V0 * (1.0 / torch.cosh(x))**2 * (1.0 + eps * torch.sin(omega * t))


def initial_psi(x):
    sigma, x0, k0 = 1.0, -5.0, 2.0
    amp = 1.0 / ((2 * np.pi * sigma**2)**0.25)
    gauss = amp * torch.exp(-(x - x0)**2 / (4 * sigma**2))
    return gauss * torch.cos(k0 * x), gauss * torch.sin(k0 * x)


# --- Training Setup ---
X_MIN, X_MAX = -10.0, 10.0
T_MIN, T_MAX = 0.0, 5.0

model = PINN([2, 64, 64, 64, 64, 2]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# --- Training ---
EPOCHS = 5000
N_COL = 600
N_IC = 400
N_BC = 200
PRINT_EVERY = 200
RESAMPLE_EVERY = 50

loss_history = []
print(f"\nTraining: {EPOCHS} epochs, {N_COL} collocation points")
print(f"  Hidden layers: [48, 48, 48]")
print("-" * 65)

start_time = time.time()

for epoch in range(EPOCHS):
    # Sample points
    if epoch % RESAMPLE_EVERY == 0:
        x_col = torch.tensor(
            np.random.uniform(X_MIN, X_MAX, (N_COL, 1)),
            dtype=torch.float32, device=DEVICE)
        t_col = torch.tensor(
            np.random.uniform(T_MIN, T_MAX, (N_COL, 1)),
            dtype=torch.float32, device=DEVICE)
        x_ic = torch.tensor(
            np.random.uniform(X_MIN, X_MAX, (N_IC, 1)),
            dtype=torch.float32, device=DEVICE)
        t_ic = torch.zeros((N_IC, 1), dtype=torch.float32, device=DEVICE)
        t_bc = torch.tensor(
            np.random.uniform(T_MIN, T_MAX, (N_BC, 1)),
            dtype=torch.float32, device=DEVICE)
        x_bc_l = X_MIN * torch.ones((N_BC, 1), dtype=torch.float32, device=DEVICE)
        x_bc_r = X_MAX * torch.ones((N_BC, 1), dtype=torch.float32, device=DEVICE)

    # Zero grad
    optimizer.zero_grad()

    # --- PDE residual ---
    x_col_g = x_col.clone().requires_grad_(True)
    t_col_g = t_col.clone().requires_grad_(True)

    out = model(x_col_g, t_col_g)
    psi_r, psi_i = out[:, 0:1], out[:, 1:2]

    # First derivatives
    ones = torch.ones_like(psi_r)
    psi_r_t = torch.autograd.grad(psi_r, t_col_g, grad_outputs=ones, create_graph=True)[0]
    psi_i_t = torch.autograd.grad(psi_i, t_col_g, grad_outputs=ones, create_graph=True)[0]
    psi_r_x = torch.autograd.grad(psi_r, x_col_g, grad_outputs=ones, create_graph=True)[0]
    psi_i_x = torch.autograd.grad(psi_i, x_col_g, grad_outputs=ones, create_graph=True)[0]

    # Second derivatives
    psi_r_xx = torch.autograd.grad(psi_r_x, x_col_g, grad_outputs=ones, create_graph=True)[0]
    psi_i_xx = torch.autograd.grad(psi_i_x, x_col_g, grad_outputs=ones, create_graph=True)[0]

    V = potential_torch(x_col_g, t_col_g)

    # PDE: i ∂ψ/∂t = -∂²ψ/∂x² + Vψ
    # Real: ∂ψ_R/∂t + ∂²ψ_I/∂x² - V·ψ_I = 0
    # Imag: ∂ψ_I/∂t - ∂²ψ_R/∂x² + V·ψ_R = 0
    f_r = psi_r_t + psi_i_xx - V * psi_i
    f_i = psi_i_t - psi_r_xx + V * psi_r
    loss_pde = torch.mean(f_r**2) + torch.mean(f_i**2)

    # --- IC loss ---
    out_ic = model(x_ic, t_ic)
    psi_r_ic, psi_i_ic = out_ic[:, 0:1], out_ic[:, 1:2]
    psi_r_true, psi_i_true = initial_psi(x_ic)
    loss_ic = torch.mean((psi_r_ic - psi_r_true)**2) + \
              torch.mean((psi_i_ic - psi_i_true)**2)

    # --- BC loss ---
    out_bc_l = model(x_bc_l, t_bc)
    out_bc_r = model(x_bc_r, t_bc)
    loss_bc = torch.mean(out_bc_l[:, 0:1]**2) + torch.mean(out_bc_l[:, 1:2]**2) + \
              torch.mean(out_bc_r[:, 0:1]**2) + torch.mean(out_bc_r[:, 1:2]**2)

    # --- Total loss ---
    loss = loss_pde + 10.0 * loss_ic + loss_bc
    loss.backward()
    optimizer.step()

    loss_history.append([loss.item(), loss_pde.item(), loss_ic.item(), loss_bc.item()])

    if epoch % PRINT_EVERY == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:4d}/{EPOCHS} | "
              f"Total: {loss.item():.2e} | PDE: {loss_pde.item():.2e} | "
              f"IC: {loss_ic.item():.2e} | BC: {loss_bc.item():.2e} | "
              f"Time: {elapsed:.1f}s")
        scheduler.step(loss)

elapsed = time.time() - start_time
print(f"\nTraining done in {elapsed:.1f}s")
print(f"Final loss: Total={loss.item():.2e}, PDE={loss_pde.item():.2e}, "
      f"IC={loss_ic.item():.2e}, BC={loss_bc.item():.2e}")

# --- Save results ---
os.makedirs('results', exist_ok=True)
torch.save(model.state_dict(), 'results/pinn_model.pt')

# Save loss history
loss_arr = np.array(loss_history)
np.savetxt('results/loss_history.csv', loss_arr,
           header='total,pde,ic,bc', delimiter=',', comments='')

# --- Predict on grid ---
print("\nGenerating predictions...")
model.eval()
nx, nt = 200, 80
x_grid = np.linspace(X_MIN, X_MAX, nx)
t_grid = np.linspace(T_MIN, T_MAX, nt)
X_mesh, T_mesh = np.meshgrid(x_grid, t_grid)
x_flat = X_mesh.flatten().reshape(-1, 1)
t_flat = T_mesh.flatten().reshape(-1, 1)

with torch.no_grad():
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=DEVICE)
    t_t = torch.tensor(t_flat, dtype=torch.float32, device=DEVICE)
    out = model(x_t, t_t)
    psi_r = out[:, 0].cpu().numpy()
    psi_i = out[:, 1].cpu().numpy()

psi_r = psi_r.reshape(nt, nx)
psi_i = psi_i.reshape(nt, nx)
prob = psi_r**2 + psi_i**2

np.savez('results/pinn_results.npz',
         x=x_grid, t=t_grid, psi_r=psi_r, psi_i=psi_i,
         prob_density=prob)
print(f"Grid: {nx} x {nt} = {nx*nt} points")

# --- Plots ---
# Loss curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(loss_arr[:, 0], 'k-', linewidth=1.5, label='Total Loss')
ax.semilogy(loss_arr[:, 1], 'b-', linewidth=1, alpha=0.7, label='PDE Loss')
ax.semilogy(loss_arr[:, 2], 'r-', linewidth=1, alpha=0.7, label='IC Loss')
ax.semilogy(loss_arr[:, 3], 'g-', linewidth=1, alpha=0.7, label='BC Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('PINN Training Loss'); ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/loss_history.png', dpi=150)
plt.close()

# Snapshots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, t_val in enumerate([0.0, 2.5, 5.0]):
    ti = np.argmin(np.abs(t_grid - t_val))
    axes[idx].plot(x_grid, prob[ti, :], 'b-', linewidth=2)
    axes[idx].set_xlabel('x'); axes[idx].set_ylabel('|psi|^2')
    axes[idx].set_title(f't = {t_grid[ti]:.2f}')
    axes[idx].set_xlim(-10, 10)
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('PINN-Predicted Wavefunction Evolution')
plt.tight_layout()
plt.savefig('results/pinn_snapshots.png', dpi=150)
plt.close()

print("\nDone! Results saved to results/")
print("=" * 65)
