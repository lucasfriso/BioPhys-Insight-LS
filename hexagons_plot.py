import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  # Set to True if you have a full LaTeX engine installed on your machine
})
# Create figure
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

# Define the wavevectors for a hexagonal pattern (120 degrees apart)
qc = 1.0  # magnitude
angles = [0, 2*np.pi/3, 4*np.pi/3]

q1 = np.array([qc * np.cos(angles[0]), qc * np.sin(angles[0])])
q2 = np.array([qc * np.cos(angles[1]), qc * np.sin(angles[1])])
q3 = np.array([qc * np.cos(angles[2]), qc * np.sin(angles[2])])

origin = np.array([0, 0])

# Plot the primary vectors
ax.quiver(*origin, q1[0], q1[1], color='r', angles='xy', scale_units='xy', scale=1, width=0.015)
ax.quiver(*origin, q2[0], q2[1], color='b', angles='xy', scale_units='xy', scale=1, width=0.015)
ax.quiver(*origin, q3[0], q3[1], color='g', angles='xy', scale_units='xy', scale=1, width=0.015)

# Draw dashed lines to show the vector addition triangle (q1 + q2 + q3 = 0)
p2 = q1 + q2
ax.plot([q1[0], p2[0]], [q1[1], p2[1]], 'b--', alpha=0.5)
ax.plot([p2[0], 0], [p2[1], 0], 'g--', alpha=0.5)

# Formatting: the marginal stability circle
circle = plt.Circle((0, 0), qc, color='k', fill=False, linestyle=':', alpha=0.3)
ax.add_patch(circle)

# Add text labels at the end of the vectors
offset = 0.15
ax.text(q1[0] + offset, q1[1], r'$\mathbf{q}_1$', fontsize=16, ha='center', va='center', color='r')
ax.text(q2[0] - offset, q2[1] + offset, r'$\mathbf{q}_2$', fontsize=16, ha='center', va='center', color='b')
ax.text(q3[0] - offset, q3[1] - offset, r'$\mathbf{q}_3$', fontsize=16, ha='center', va='center', color='g')

# Add the 120 degree arcs
arc1 = mpatches.Arc(origin, 0.4, 0.4, angle=0, theta1=0, theta2=120, color='k', linewidth=1.5)
ax.add_patch(arc1)
ax.text(0.15, 0.25, r'$120^\circ$', fontsize=14)

arc2 = mpatches.Arc(origin, 0.4, 0.4, angle=0, theta1=120, theta2=240, color='k', linewidth=1.5)
ax.add_patch(arc2)
ax.text(-0.35, 0.0, r'$120^\circ$', fontsize=14)

arc3 = mpatches.Arc(origin, 0.4, 0.4, angle=0, theta1=240, theta2=360, color='k', linewidth=1.5)
ax.add_patch(arc3)
ax.text(0.15, -0.25, r'$120^\circ$', fontsize=14)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Wavevector Triad for Hexagonal Patterns\n" + r"$\mathbf{q}_1 + \mathbf{q}_2 + \mathbf{q}_3 = 0$", fontsize=16, pad=20)

plt.tight_layout()
plt.savefig('hexagonal_triad.pdf', format='pdf', bbox_inches='tight')
plt.show()