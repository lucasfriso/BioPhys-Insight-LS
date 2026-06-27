import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if you have a full LaTeX engine installed on your machine
})

# Create the spatial grid
x = np.linspace(-10, 10, 800)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# Base critical wavenumber
k0 = 2.0

# --- 1. Eckhaus Instability (Longitudinal phase modulation) ---
# We modulate the phase along the x-axis to show compression and dilation.
# Mathematically: A(X) ~ cos(k0 * X + phi(X))
phase_eckhaus = k0 * X + 2.5 * np.sin(0.4 * X)

# Apply a smooth thresholding (tanh) to mimic a saturated reaction-diffusion pattern
Z_eck = np.tanh(2.0 * np.cos(phase_eckhaus))

# --- 2. Zigzag Instability (Transverse phase modulation) ---
# We modulate the phase along the y-axis to show transverse buckling.
# Mathematically: A(X, Y) ~ cos(k0 * X + phi(Y))
phase_zigzag = k0 * X + 1.5 * np.sin(1.5 * Y)

# Apply the same non-linear thresholding
Z_zig = np.tanh(2.0 * np.cos(phase_zigzag))

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(6, 6), dpi=150)

# Plot Eckhaus
axes[0].imshow(Z_eck, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='bone')
axes[0].set_title('Eckhaus Instability\n(Longitudinal Compression / Dilation)',
                  fontsize=14, pad=10)
axes[0].axis('off')

# Plot Zigzag
axes[1].imshow(Z_zig, extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', cmap='bone')
axes[1].set_title('Zigzag Instability\n(Transverse Buckling)',
                  fontsize=14, pad=10)
axes[1].axis('off')

plt.tight_layout()

# Save the figure to be used directly in the LaTeX document
plt.savefig('eckhaus_zigzag_snapshots.pdf', bbox_inches='tight')
plt.show()