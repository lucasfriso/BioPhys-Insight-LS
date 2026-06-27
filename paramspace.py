import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ---------------------------------------------------------
# Plot Configuration
# ---------------------------------------------------------
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if you have a full LaTeX engine installed on your machine
})

# ---------------------------------------------------------
# Mathematical Setup & Linear Stability Analysis
# ---------------------------------------------------------
Du = 2e-5
Dv = 0.5e-5

# We can keep resolution high because contourf handles it efficiently
resolution = 500
F_vals = np.linspace(0.001, 0.09, resolution)
k_vals = np.linspace(0.001, 0.09, resolution)
F_grid, k_grid = np.meshgrid(F_vals, k_vals)

state = np.zeros_like(F_grid, dtype=int)

for i in range(resolution):
    for j in range(resolution):
        F = F_grid[i, j]
        k = k_grid[i, j]

        discriminant = 1.0 - 4.0 * (F + k) ** 2 / F
        if discriminant < 0:
            continue

        u = 0.5 - 0.5 * np.sqrt(discriminant)
        v = (F + k) / u

        J11 = -v ** 2 - F
        J12 = -2 * u * v
        J21 = v ** 2
        J22 = 2 * u * v - (F + k)

        trace = J11 + J22
        det = J11 * J22 - J12 * J21

        if det < 0:
            continue

        if trace > 0:
            state[i, j] = 2
        else:
            H = Dv * J11 + Du * J22
            if H > 0 and H ** 2 > 4 * Du * Dv * det:
                state[i, j] = 3
            else:
                state[i, j] = 1

            # ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#e0f3f8', '#ccece6', '#fff2cc', '#f4a582']
cmap = ListedColormap(colors)

# FIX: Use contourf instead of pcolormesh.
# We set levels slightly offset from integers to properly capture the 0, 1, 2, 3 states.
levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
c = ax.contourf(F_grid, k_grid, state, levels=levels, colors=colors)

# Draw boundary lines for crisp edges
ax.contour(F_grid, k_grid, state, levels=[0.5, 1.5, 2.5], colors='black', linewidths=1.2)

# Formatting
ax.set_title(r'Linear Stability Analysis in the $F-k$ Plane')
ax.set_xlabel(r'Feed Rate parameter $F$')
ax.set_ylabel(r'Kill Rate parameter $k$')
ax.set_xlim(0.001, 0.09)
ax.set_ylim(0.001, 0.09)

# Create a custom legend
labels = [
    'Dead State (Trivial only)',
    'Stable Homogeneous',
    'Hopf Instability (Oscillations)',
    'Turing Instability (Patterns)'
]
patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]
ax.legend(handles=patches, loc='upper right', framealpha=0.95, edgecolor='black')

plt.tight_layout()

# Save as PDF (will now be very lightweight)
plt.savefig('grayscott_bifurcation_lightweight.pdf', format='pdf', bbox_inches='tight')
plt.show()