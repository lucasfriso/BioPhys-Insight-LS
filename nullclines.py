import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True  #
})


# 1. Define Kinetic Parameters
# (These specific values place the fixed point in the unstable regime,
# resulting in a limit cycle)
a = 0.08
b = 0.6

# 2. Create the domain for x (Product concentration)
# Avoid exactly 0 to prevent division by zero in edge cases, though a>0 protects us here.
x = np.linspace(0, 3, 500)

# 3. Define the Nullcline Equations
y_xdot = x / (a + x**2)  # dx/dt = 0 nullcline
y_ydot = b / (a + x**2)  # dy/dt = 0 nullcline

# 4. Calculate Analytical Points of Interest
# The Fixed Point (Intersection)
x_star = b
y_star = b / (a + b**2)

# The Local Maximum of the x-nullcline
x_max = np.sqrt(a)
y_max = 1 / (2 * np.sqrt(a))

# The Inflection Point of the x-nullcline
x_inf = np.sqrt(3 * a)
y_inf = x_inf / (a + x_inf**2)

# 5. Plotting
plt.figure(figsize=(10, 6))

# Plot the curves
plt.plot(x, y_xdot, lw=2, color='lightcoral', label=r'$\dot{x}=0$ Nullcline: $y = \frac{x}{a+x^2}$')
plt.plot(x, y_ydot, lw=2, color='lightblue', label=r'$\dot{y}=0$ Nullcline: $y = \frac{b}{a+x^2}$')

# Mark the Fixed Point
plt.plot(x_star, y_star, 'ko', markersize=8, zorder=5,
         label=f'Fixed Point ($x^*={x_star:.2f}, y^*={y_star:.2f}$)')

# Mark the Local Maximum
plt.plot(x_max, y_max, 'go', markersize=8, zorder=5,
         label=f'Local Max ($x={x_max:.2f}, y={y_max:.2f}$)')

# Mark the Inflection Point
plt.plot(x_inf, y_inf, 'mo', markersize=6, zorder=5,
         label=f'Inflection Point ($x={x_inf:.2f}$)')

# Formatting and aesthetics
plt.title('Nullcline Geometry of the Selkov Model', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 3)
plt.ylim(0, np.max(y_xdot) + 0.2)

plt.tight_layout()
plt.savefig('Nullcline.pdf')
plt.show()