import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Plot Configuration for Publication
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if using LaTeX
})

# ==========================================
# Mathematical Setup
# ==========================================
epsilon = 0.1         # The scale separation parameter
q_c = 2.0             # Fast spatial frequency

# Define the physical 2D grid
x = np.linspace(-60, 60, 500)
y = np.linspace(-60, 60, 500)
x_grid, y_grid = np.meshgrid(x, y)

# Define the slow spatial coordinates (NWS scaling)
# X = epsilon * x
# Y = sqrt(epsilon) * y
X_grid = epsilon * x_grid
Y_grid = np.sqrt(epsilon) * y_grid

# 1. The Slow Envelope A(X, Y)
# Using a 2D Gaussian wave packet to represent the localized amplitude
A_XY = 1.5 * np.exp(-(X_grid**2 + Y_grid**2) / 10.0)

# 2. The Fast Carrier Wave (Stripes along x)
carrier = np.cos(q_c * x_grid)

# 3. The Combined Physical Ansatz
ansatz = A_XY * carrier

# ==========================================
# Plotting
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the 2D field using a contour map
# We use a colormap that clearly shows positive (red) and negative (blue) fluctuations
c = ax.pcolormesh(x_grid, y_grid, ansatz, cmap='RdBu_r', shading='auto', vmin=-1.5, vmax=1.5)

# Add a colorbar to represent the concentration fluctuation \tilde{u}
cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Fluctuation $\tilde{u}$', rotation=270, labelpad=20)

# ------------------------------------------
# Annotations to explain the scales
# ------------------------------------------
# 1. Annotate the Fast Wavelength (x-direction)


# 2. Annotate the Slow Transverse Scale (y-direction)
ax.annotate('', xy=(40, -20), xytext=(40, 20),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(42, 0, r'$Y \sim \sqrt{\epsilon}y$', ha='left', va='center', color='black', fontsize=12, backgroundcolor='white')

# 3. Annotate the Slow Longitudinal Scale (x-direction)
ax.annotate('', xy=(-30, 45), xytext=(30, 45),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0, 47, r'$X \sim \epsilon x$', ha='center', va='bottom', color='black', fontsize=12, backgroundcolor='white')

# Formatting
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('stripe_ansatz_2d.pdf', format='pdf', bbox_inches='tight')
plt.show()