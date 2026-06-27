import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. System Parameters
# ==========================================
Du, Dv = 0.16, 0.08
k = 0.0616
dt = 1.0

# --- SIMULATION STATE (Pushed into instability) ---
F = 0.053
u_s, v_s = 0.45305, 0.25295

# --- CRITICAL STATE (From your derivation) ---
F_c = 0.054975
u_c = 0.447069
v_c = 0.260754
q_c = 0.586788

# --- YOUR ANALYTICAL DERIVATIONS ---
analytical_coeffs = {
    'xi_0_sq': 0.198,
    'g': 8.9
}


# ==========================================
# 2. Mathematical Normalization (The Fredholm Fix)
# ==========================================
def compute_critical_eigenvectors():
    """Calculates the exact Left Eigenvector (psi) normalized to your paper."""
    # 1. Jacobian at the critical point
    fu = -v_c ** 2 - F_c
    fv = -2 * u_c * v_c
    gu = v_c ** 2
    gv = 2 * u_c * v_c - (F_c + k)

    J_c = np.array([[fu, fv], [gu, gv]])
    Diff = np.array([[Du, 0], [0, Dv]])

    # 2. Critical Linear Operator L = J - q_c^2 * D
    L_c = J_c - (q_c ** 2) * Diff

    # 3. Right Eigenvector (rho) in the nullspace.
    # Normalized so U-component = 1 (Matching your algebra!)
    rho = np.array([1.0, -L_c[0, 0] / L_c[0, 1]])

    # 4. Left Eigenvector (psi) in the transposed nullspace
    psi_raw = np.array([-L_c[1, 1] / L_c[0, 1], 1.0])

    # 5. The Adjoint Normalization constraint: <psi, rho> = 1
    psi = psi_raw / np.dot(psi_raw, rho)

    return psi


# Calculate the left eigenvector for amplitude extraction
psi = compute_critical_eigenvectors()
print(f"--- Analytical Normalization ---")
print(f"Calculated Left Eigenvector Psi = [{psi[0]:.4f}, {psi[1]:.4f}]\n")

# ==========================================
# 3. Box Matching & Simulator Setup
# ==========================================
n_waves = 11
L_exact = n_waves * (2 * np.pi / q_c)
N = 128
dx = L_exact / N


def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / (dx ** 2)


def run_simulation(q_target, steps, record_every=10):
    U = np.ones((N, N)) * u_s
    V = np.ones((N, N)) * v_s

    x = np.arange(N) * dx
    X, _ = np.meshgrid(x, x)

    perturbation = 0.0001 * np.cos(q_target * X)
    U += perturbation
    V += perturbation

    time_array, amp_array = [], []
    for step in range(steps):
        uvv = U * V ** 2
        U += dt * (Du * laplacian(U) - uvv + F * (1 - U))
        V += dt * (Dv * laplacian(V) + uvv - (F + k) * V)

        if step % record_every == 0:
            # THE AMPLITUDE FIX: Project the full state onto the left eigenvector
            U_fluct = U - np.mean(U)
            V_fluct = V - np.mean(V)

            # W = psi_1 * U + psi_2 * V (Exact Fredholm projection)
            W_field = psi[0] * U_fluct + psi[1] * V_fluct

            W_fft = np.fft.fft2(W_field) / (N * N)
            amp = np.max(np.abs(W_fft)) * 2.0

            time_array.append(step * dt)
            amp_array.append(amp)

    return np.array(time_array), np.array(amp_array)


# ==========================================
# 4. Numerical Fitting Functions
# ==========================================
def linear_growth_func(t, A0, mu_sigma):
    return A0 * np.exp(mu_sigma * t)


def nws_parabola_func(K, mu_sigma_fixed, xi_sq, K_shift):
    # THE DISPERSION FIX: Explicitly include the 4*q_c^2 operator factor
    return mu_sigma_fixed - xi_sq * (4 * (q_c ** 2)) * ((K - K_shift) ** 2)


# ==========================================
# 5. Core Extraction Execution
# ==========================================
extracted_data = {}

print(f"--- Phase 1: Extracting Growth and Saturation at q_c = {q_c:.5f} ---")
t, A = run_simulation(q_target=q_c, steps=5000, record_every=50)

t_early, A_early = t[:15], A[:15]
popt, _ = curve_fit(linear_growth_func, t_early, A_early, p0=[A[0], 0.005])
mu_sigma_num = popt[1]

A_sat = np.mean(A[-20:])
g_num = mu_sigma_num / (A_sat ** 2)
extracted_data['g'] = g_num

print(f"-> Numerically measured mu*sigma: {mu_sigma_num:.6f}")
print(f"-> Numerically measured Projected A_sat: {A_sat:.6f}")

print("\n--- Phase 2: Extracting Dispersion Curvature xi_0^2 ---")
K_vals = np.linspace(-0.03, 0.03, 7)
lambdas = []
for K in K_vals:
    q_test = q_c + K
    t_disp, A_disp = run_simulation(q_target=q_test, steps=800, record_every=20)
    popt_disp, _ = curve_fit(linear_growth_func, t_disp, A_disp, p0=[A_disp[0], 0.005])
    lambdas.append(popt_disp[1])

# Fit using the NWS specific parabola
popt_para, _ = curve_fit(
    lambda K, xi_sq, K_shift: nws_parabola_func(K, mu_sigma_num, xi_sq, K_shift),
    K_vals, lambdas, p0=[0.1, 0.0]
)
xi_sq_num, k_shift_num = popt_para
extracted_data['xi_0_sq'] = xi_sq_num

# ==========================================
# 6. Final Comparison Table
# ==========================================
print("\n=======================================================")
print(f"{'Coefficient':<15} | {'Analytical':<12} | {'Numerical':<12} | {'% Error':<10}")
print("-------------------------------------------------------")

for key in analytical_coeffs:
    ana = analytical_coeffs[key]
    num = extracted_data[key]
    err = abs((num - ana) / ana) * 100
    print(f"{key:<15} | {ana:<12.5f} | {num:<12.5f} | {err:<8.2f}%")
print("=======================================================")