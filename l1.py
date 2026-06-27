import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True
})

# 1. Define the interval for x = omega^2
# The physical parameter constraint requires 0 < omega^2 < 1
omega_sq = np.linspace(-2, 2, 2000)

# 2. Define the first Lyapunov coefficient function
l1 = (2 * omega_sq**2 - 2 * omega_sq - 1) / 8

# 3. Create the plot
plt.figure(figsize=(8, 5))
plt.plot(omega_sq, l1, label=r"$l_1(\omega^2)$", color="royalblue", linewidth=2.5)

# 4. Highlight the physical bounds and key features
# Draw a dashed line at l1 = 0 to show it's strictly below the axis
plt.axhline(0, color="crimson", linestyle="--", linewidth=1.2)
plt.axvline(0,color="crimson", linestyle="--", linewidth=1.2)
plt.axvline(1, color="crimson", linestyle="--", linewidth=1.2)
# Highlight the vertex (minimum) at omega^2 = 0.5
plt.plot(0.5, -1.5/8, 'go')

# 5. Styling the graph
plt.title("$l_1$ vs. $\omega^2$ in the Sel'kov Model", fontsize=12, pad=15)
plt.xlabel(r"$\omega^2$", fontsize=11)
plt.ylabel(r"$l_1$ ", fontsize=11)
plt.xlim(-1, 2)
plt.ylim(-1, 1)  # Zoomed in to clearly show it stays negative

# Add a fill to visually emphasize the negative region
plt.fill_between(omega_sq, l1, 0, where=(l1 < 0), color="royalblue", alpha=0.1)

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

# Show the plot
plt.tight_layout()
plt.savefig('L1.pdf')
plt.show()