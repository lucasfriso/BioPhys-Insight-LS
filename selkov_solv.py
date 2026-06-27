import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'text.usetex': True
})


# =============================================================================
# 1. PARAMETERS & SYSTEM DEFINITION
# =============================================================================
a = 0.124
b = 0.6

# The Modified Selkov ODEs
def selkov(t, state, a, b):
    x, y = state
    dxdt = -x + a*y + (x**2)*y
    dydt = b - a*y - (x**2)*y
    return [dxdt, dydt]

# Calculate the fixed point analytically
x_star = b
y_star = b / (a + b**2)

# =============================================================================
# 2. PERIOD COMPARISON: NUMERICAL VS ANALYTICAL
# =============================================================================

# --- A. NUMERICAL PERIOD EXTRACTION ---
# Run a long, high-resolution simulation so the system settles on the limit cycle
t_span_long = (0, 1000)
t_eval_long = np.linspace(0, 1000, 50000) # High res for accurate peak times
sol_long = solve_ivp(selkov, t_span_long, [0.1, 0.1], args=(a, b), t_eval=t_eval_long, method='RK45')

# Find peaks in the x(t) time series
peaks, _ = find_peaks(sol_long.y[0])

# Discard the first 10 peaks to ensure transient behavior is completely gone
steady_peaks = peaks[10:]
peak_times = sol_long.t[steady_peaks]

# The numerical period is the average time between consecutive steady peaks
T_num = np.mean(np.diff(peak_times))

# --- B. ANALYTICAL PERIOD ESTIMATION (LINDSTEDT-POINCARE) ---
# 1. Find a_c (the exact value of 'a' at the Hopf bifurcation for our 'b')
# Trace = 0 implies: a^2 + (2*b^2 + 1)*a + (b^4 - b^2) = 0
coef_A = 1.0
coef_B = 2*b**2 + 1
coef_C = b**4 - b**2
a_c = (-coef_B + np.sqrt(coef_B**2 - 4*coef_A*coef_C)) / (2*coef_A)

# 2. Calculate Unperturbed Frequency (omega_c) and Small Parameter (tau)
omega_c = np.sqrt(a_c + b**2)
tau = (b**2 - a)/(a + b**2) - (a + b**2) # Trace at our actual 'a'

# 3. Calculate Amplitude Damping Constant (K)
K = 0.75 + (b**2 * (1 - 2*omega_c**2)) / (2 * omega_c**4)

# 4. Calculate Frequency Correction (omega_2)
term1 = 1 / (4 * omega_c)
term2 = b**2 / (3 * omega_c**3)
term3 = (b**2 * (1 - 2*omega_c**2)**2) / (3 * omega_c**5)
omega_2 = (tau / (2 * K)) * (term1 - term2 - term3)

# 5. Final Analytical Period
omega_approx = omega_c + omega_2
T_ana = 2 * np.pi / omega_approx
T_ananc=2*np.pi/omega_c

# --- C. PRINT RESULTS ---
print("==================================================")
print(f" SELKOV MODEL PERIOD ANALYSIS (a={a}, b={b})")
print("==================================================")
print(f"Bifurcation Point a_c : {a_c:.5f}")
print(f"Linear Trace (tau)    : {tau:.5f}\n")
print(f"Numerical Period      : {T_num:.5f} time units")
print(f"Analytical Period     : {T_ana:.5f} time units")
print(f"Analytical Period   non corr  : {T_ananc:.5f} time units")
error = abs(T_num - T_ana) / T_num * 100
errornc = abs(T_num - T_ananc) / T_num * 100
print(f"Percentage Error      : {error:.2f}%")
print(f"Percentage Error   non corr   : {errornc:.2f}%")
print("==================================================")


# =============================================================================
# 3. FIGURE 1: PHASE PLANE & POINCARE-BENDIXSON TRAPPING REGION
# =============================================================================
plt.figure(figsize=(8, 6))

X, Y = np.meshgrid(np.linspace(0, 2.5, 20), np.linspace(0, 9, 20))
u = -X + a*Y + (X**2)*Y
v = b - a*Y - (X**2)*Y
plt.streamplot(X, Y, u, v, color='lightgray', density=1.2, linewidth=1, arrowsize=1)

t_span = (0, 100)
t_eval = np.linspace(0, 100, 2000)
initial_conditions = [(0.1, 0.1),(1,1)]

for ic in initial_conditions:
    sol = solve_ivp(selkov, t_span, ic, args=(a, b), t_eval=t_eval, method='RK45')
    plt.plot(sol.y[0], sol.y[1], linewidth=1.5, label=f'Trajectory from {ic}')

plt.plot(x_star, y_star, 'ro', markersize=8, label='Unstable Fixed Point')

x_flat = np.linspace(0, b, 100)
y_flat = np.full_like(x_flat, b/a)
C = b + (b/a)
x_diag = np.linspace(b, 2.5, 100)
y_diag = C - x_diag

plt.plot(x_flat, y_flat, 'k--', linewidth=2, label='Horizontal Roof ($y = b/a$)')
plt.plot(x_diag, y_diag, 'k-', linewidth=2, label='Diagonal Roof ($x+y=C$)')

plt.xlim(-0.01, 2)
plt.ylim(-0.01, 9)
plt.xlabel('Product $x$', fontsize=14)
plt.ylabel('Substrate $y$', fontsize=14)
plt.title('Modified Selkov Phase Plane \& Trapping Region', fontsize=16)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('phase_plane.pdf')

# =============================================================================
# 4. FIGURE 2: THE a-b PARAMETER PLANE (HOPF BIFURCATION)
# =============================================================================
plt.figure(figsize=(8, 6))

A_vals = np.linspace(0.01, 0.2, 500)
B_vals = np.linspace(0.1, 1.2, 500)
A, B = np.meshgrid(A_vals, B_vals)

Trace = (B**2 - A) / (A + B**2) - (A + B**2)

plt.contourf(A, B, Trace, levels=[-np.inf, 0], colors=['lightblue'], alpha=0.5)
plt.contourf(A, B, Trace, levels=[0, np.inf], colors=['lightcoral'], alpha=0.5)
plt.contour(A, B, Trace, levels=[0], colors='k', linewidths=2)

plt.plot(a, b, 'k*', markersize=12, label=f'Simulation Point\n(a={a}, b={b})')

plt.text(0.16, 0.6, 'Stable Steady State\n(Dead Cell)', fontsize=12, ha='center')
plt.text(0.04, 0.6, 'Limit Cycle\n(Oscillations)', fontsize=12, ha='center')

plt.xlabel('Basal Rate parameter $a$', fontsize=14)
plt.ylabel('Feed Rate parameter $b$', fontsize=14)
plt.title('Hopf Bifurcation in the Parameter Plane', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('bifurcation.pdf')

plt.show()