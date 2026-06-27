import numpy as np
import matplotlib.pyplot as plt

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

# Define the Q axis (deviation from critical wavenumber)
Q = np.linspace(-1.5, 1.5, 500)

# Define the curves based on the NWS amplitude equation boundaries
# Marginal Stability Curve (Neutral curve where A = 0)
# mu*sigma = xi_0^2 * Q^2  --> We plot scaled mu = Q^2
mu_neutral = Q**2

# Eckhaus Instability Curve (where D_parallel = 0)
# mu*sigma = 3 * xi_0^2 * Q^2 --> We plot scaled mu = 3*Q^2
mu_eckhaus = 3 * Q**2

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the bounding curves
ax.plot(Q, mu_neutral, 'k-', linewidth=2, label='Neutral Stability Curve')
ax.plot(Q, mu_eckhaus, 'r--', linewidth=2, label='Eckhaus Boundary')
ax.axvline(0, color='b', linestyle='-.', linewidth=2, label='Zigzag Boundary ($Q=0$)')

# Shade the Stable Busse Balloon (Bounded by Eckhaus and Zigzag)
# Stable region is where Q > 0 AND mu > 3*Q^2
Q_stable = np.linspace(0, 1.5, 250)
mu_stable = 3 * Q_stable**2
ax.fill_between(Q_stable, mu_stable, 6, color='#ccece6', alpha=0.8, label='Stable Stripe Pattern')

# Shade the unstable regions for clarity
ax.fill_between(Q, mu_neutral, np.where(Q > 0, np.maximum(mu_neutral, mu_eckhaus), 6),
                color='#f4a582', alpha=0.3) # Eckhaus Unstable Region
ax.fill_between(np.linspace(-1.5, 0, 250), np.linspace(0, 0, 250)**2, 6,
                color='#fff2cc', alpha=0.5) # Zigzag Unstable Region

# Annotations
ax.text(0.5, 3, 'Stable\nStripes', ha='center', va='center', fontsize=14, fontweight='bold', color='darkgreen')
ax.text(-0.75, 3, 'Zigzag\nInstability', ha='center', va='center', fontsize=12, color='darkgoldenrod')
ax.text(1.15, 2.5, 'Eckhaus\nInstability', ha='center', va='center', fontsize=12, color='darkred', rotation=65)

# Formatting
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 6.0)
ax.set_xlabel(r'$Q = q - q_c$',loc='right')
ax.set_ylabel(r'$\mu\sigma / \xi_0^2$',loc='top')
ax.set_title('Stability for Stripe Patterns')
ax.legend(loc='upper right', framealpha=0.95, edgecolor='black',prop={'size': 10})

# Clean axes (cross at 0,0)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.xaxis.set_label_coords(1.0, -0.01)
ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('stripe_stability_diagram.pdf', format='pdf', bbox_inches='tight')
plt.show()