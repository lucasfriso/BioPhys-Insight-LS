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



# ---------------------------------------------------------

# Mathematical Setup

# ---------------------------------------------------------

q = np.linspace(0.1, 1.9, 500)

qc = 1.0       # Critical wavevector

xi = 1.0       # Curvature coefficient



# Define the control parameter epsilon for the three regimes

eps_super = 0.15   # F > F_c (Unstable regime)

eps_crit = 0.0     # F = F_c (Critical regime)

eps_sub = -0.15    # F < F_c (Stable regime)



# Calculate the eigenvalues

sigma_super = eps_super - xi * (q - qc)**2

sigma_crit = eps_crit - xi * (q - qc)**2

sigma_sub = eps_sub - xi * (q - qc)**2



# Calculate q1 and q2 for the supercritical case

q1 = qc - np.sqrt(eps_super / xi)

q2 = qc + np.sqrt(eps_super / xi)



# ---------------------------------------------------------

# Plotting

# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))



# Plot the three curves

ax.plot(q, sigma_super, 'k-', linewidth=2, label=r'$F > F_c$ (Supercritical)')

ax.plot(q, sigma_crit, 'k--', linewidth=2, label=r'$F = F_c$ (Critical)')

ax.plot(q, sigma_sub, 'k-.', linewidth=2, label=r'$F < F_c$ (Subcritical)')



# Axes and layout

ax.axhline(0, color='black', linewidth=1) # x-axis

ax.axvline(0, color='black', linewidth=1) # y-axis



# Annotations

ax.plot(qc, eps_super, 'k+', markersize=10) # Peak

ax.text(qc, eps_super + 0.02, r'$q_c$', ha='center', va='bottom', fontsize=14)



# Mark q1 and q2

ax.plot([q1, q2], [0, 0], 'ro', markersize=6)

ax.text(q1, -0.03, r'$q_1$', ha='center', va='top', fontsize=14, color='red')

ax.text(q2, -0.03, r'$q_2$', ha='center', va='top', fontsize=14, color='red')



# Bandwidth arrow (Epsilon / unstable band)

ax.annotate('', xy=(q1, 0.08), xytext=(q2, 0.08),

            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))

ax.text(qc, 0.09, r'Unstable Band', ha='center', va='bottom', fontsize=12, color='blue')



# Formatting

ax.set_xlabel(r'Wavevector $q$', loc='right')

ax.set_ylabel(r'Growth rate $\sigma_q^+$', loc='top')

ax.set_xlim(-0.1, 2.0)

ax.set_ylim(-0.3, 0.25)

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_xticks([])

ax.set_yticks([])

ax.legend(loc='upper right', frameon=False)



plt.tight_layout()

plt.savefig('dispersion_relation.pdf', format='pdf', bbox_inches='tight')

plt.show()