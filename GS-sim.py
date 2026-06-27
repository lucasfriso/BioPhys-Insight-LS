import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.fft as fft

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if you have a full LaTeX engine installed on your machine
})

# ==========================================================
# 1. System Parameters (Supercritical Regime)
# ==========================================================
# Base physical parameters
Du, Dv = 0.48, 0.08
k = 0.05

# Critical Bifurcation Point (Analytically Derived)
Fc = 0.0454714
qc = 0.652647
uc, vc = 0.277404, 0.344160
phi_u = 1.0

# WNLA Output Coefficients
sigma = -0.588089
xi_0_sq = 0.213129
g = 2.7979
V00_u = 4.04865  # First component of the V00 vector from WNLA

# Penetration parameters
epsilon = 0.1
mu = -1.0  # MUST be negative so that mu*sigma > 0
F = Fc + (epsilon ** 2) * mu

# ==========================================================
# 2. Grid Setup (Accounting for Anisotropic Scaling)
# ==========================================================
# Physical grid (Fast variables)
Lx, Ly = 150.0, 150.0
Nx, Ny = 256, 256
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
xx, yy = np.meshgrid(x, y, indexing='ij')

# Modulated grid (Slow variables: X = eps*x, Y = sqrt(eps)*y)
X = epsilon * xx
Y = np.sqrt(epsilon) * yy

dt = 0.05
T_final = 500  # Physical time to integrate

# ==========================================================
# 3. Initialization: 2D Localized Wave Packet
# ==========================================================
# Create a 2D Gaussian envelope in the center of the domain
X0, Y0 = epsilon * (Lx / 2), np.sqrt(epsilon) * (Ly / 2)
# Keep initial amplitude small to prevent transient PDE shock
A_initial = 0.005 * np.exp(-((X - X0) ** 2 + (Y - Y0) ** 2) / 10.0)

# Full PDE state: Homogeneous + (Envelope * Carrier Wave)
# Re(A * e^{iqx}) = 0.5 * (A * e^{iqx} + A^* * e^{-iqx})
u = uc + epsilon * 2.0 * np.real(A_initial * np.exp(1j * qc * xx)) * phi_u
v = vc + np.zeros_like(u)  # Forcing perturbation strictly on u

# NWS initial state
A = A_initial.astype(complex)

# ==========================================================
# 4. Spectral Setup for NWS Equation
# ==========================================================
dX, dY = epsilon * dx, np.sqrt(epsilon) * dy
K_X = 2 * np.pi * fft.fftfreq(Nx, d=dX)
K_Y = 2 * np.pi * fft.fftfreq(Ny, d=dY)
KX, KY = np.meshgrid(K_X, K_Y, indexing='ij')

# The spectral representation of the NWS spatial operator: (d_X - i/(2qc) d_Y^2)^2
L_op = - (KX + KY ** 2 / (2 * qc)) ** 2

dT = (epsilon ** 2) * dt  # Slow time step

print("Integrating 2D System...")
for step in range(int(T_final / dt)):
    # --- 1. Gray-Scott Full PDE (Finite Difference) ---
    lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2 * u) / dx ** 2 + \
            (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2 * u) / dy ** 2

    lap_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2 * v) / dx ** 2 + \
            (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2 * v) / dy ** 2

    uv2 = u * v ** 2
    u += (Du * lap_u - uv2 + F * (1 - u)) * dt
    v += (Dv * lap_v + uv2 - (F + k) * v) * dt

    # --- 2. NWS Amplitude Equation (Pseudo-Spectral Split-Step) ---
    A_hat = fft.fft2(A)
    A_hat = A_hat * np.exp((mu * sigma + xi_0_sq * L_op) * dT)
    A = fft.ifft2(A_hat)

    A += (- g * np.abs(A) ** 2 * A) * dT

print("Integration Complete.")

# ==========================================================
# 5. Data Extraction & Plotting (Inverted Triangle Layout)
# ==========================================================
# Make the figure taller to accommodate two rows
fig = plt.figure(figsize=(12, 10))

# Create a 2x2 grid, but we will make the bottom plot span both columns
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.3)

# ----------------------------------------------------------
# Plot 1: Full PDE 2D Field (Top Left)
# ----------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(u.T - uc, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
ax1.set_title(r"Gray-Scott PDE $(u - u_c)$", fontsize=14, pad=10)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
plt.colorbar(im1, ax=ax1, shrink=0.85)

# ----------------------------------------------------------
# Plot 2: Reconstructed Pattern from NWS (Top Right)
# ----------------------------------------------------------
reconstructed = epsilon * 2.0 * np.real(A * np.exp(1j * qc * xx)) * phi_u
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(reconstructed.T, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdBu_r')
ax2.set_title(r"NWS", fontsize=14, pad=10)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
plt.colorbar(im2, ax=ax2, shrink=0.85)

# ----------------------------------------------------------
# Plot 3: 1D Cross-Section Overlay (Bottom Row, spans both columns)
# ----------------------------------------------------------
ax3 = fig.add_subplot(gs[1, :])  # The ':' means it spans all columns in row 1
mid_idx = Ny // 2
ax3.plot(x, u[:, mid_idx] - uc, 'k-', alpha=0.7, label=r'PDE')

# 1. Calculate the linear base-state shift (The exact physics of changing F_c to F_op)
F_op = Fc + (epsilon ** 2) * mu
disc_op = 1 - 4 * (F_op + k)**2 / F_op
vc_op = F_op / (2 * (F_op + k)) * (1 + np.sqrt(disc_op))
uc_op = (F_op + k) / vc_op

linear_shift = uc_op - uc

# 2. Calculate the nonlinear mean-flow shift
nonlinear_shift = (epsilon**2) * 2.0 * (np.abs(A[:, mid_idx])**2) * V00_u

# 3. Combine them for the total theoretical baseline shift
baseline_shift = linear_shift + nonlinear_shift

# 4. Apply to envelope bounds
envelope_base = epsilon * 2.0 * np.abs(A[:, mid_idx]) * phi_u
envelope_upper = baseline_shift + envelope_base
envelope_lower = baseline_shift - envelope_base

ax3.plot(x, envelope_upper, 'r--', lw=2.5, label=r'NWS Envelope $+\mathcal{O}(\epsilon^2)$ Shift')
ax3.plot(x, envelope_lower, 'r--', lw=2.5)

ax3.set_title("1D Cross-Section Overlay", fontsize=14, pad=10)
ax3.set_xlabel(r"$x$", fontsize=12)
ax3.set_ylabel(r"Concentration Fluctuation $(u - u_c)$", fontsize=12)
ax3.legend(frameon=False, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('NWS_Envelope.pdf', bbox_inches='tight')
plt.show()