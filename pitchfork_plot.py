import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if you have a full LaTeX engine installed on your machine
})

mu_stable = np.linspace(0, 4, 400)
mu_unstable_neg = np.linspace(-2, 0, 200)
mu_unstable_pos = np.linspace(0, 4, 200)

# Calculate Amplitude |A| = sqrt(mu * sigma / g). Assuming sigma/g = 1 for normalized plot.
A_stable_pos = np.sqrt(mu_stable)
A_stable_neg = -np.sqrt(mu_stable)

fig, ax = plt.subplots(figsize=(7, 5))

# Plot the branches
ax.plot(mu_stable, A_stable_pos, 'b-', linewidth=2.5, label='Stable Pattern ($|A| \propto \sqrt{\mu}$)')
ax.plot(mu_stable, A_stable_neg, 'b-', linewidth=2.5)
ax.plot(mu_unstable_neg, np.zeros_like(mu_unstable_neg), 'b-', linewidth=2.5, label='Stable Homogeneous ($A=0$)')
ax.plot(mu_unstable_pos, np.zeros_like(mu_unstable_pos), 'r--', linewidth=2.5, label='Unstable Homogeneous')

# Formatting
ax.axvline(0, color='k', linestyle=':', linewidth=1.5)
ax.set_xlabel(r'Bifurcation Parameter $\mu \propto (F - F_c)$')
ax.set_ylabel(r'Steady-State Amplitude $|A|$')
ax.set_title('Supercritical Pitchfork Bifurcation')
ax.set_xticks([0])
ax.set_xticklabels(['$\mu_c = 0$'])
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(framealpha=0.95, edgecolor='black', loc='upper left',prop={'size': 11})

plt.tight_layout()
plt.savefig('pitchfork_bifurcation.pdf', format='pdf')
plt.show()