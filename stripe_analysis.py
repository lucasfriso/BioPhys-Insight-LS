import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. System Parameters & Analytical Inputs
# ==========================================
Du, Dv = 0.16, 0.08
F, k = 0.022, 0.051
dt = 1.0
dx = 1.0
N = 128

# --- YOUR ANALYTICAL VALUES GO HERE ---
# The exact Homogeneous Steady State (HSS) for your F and k
u_s = 0.41184  # Replace with exact analytical u*
v_s = 0.17725  # Replace with exact analytical v*
# The critical wavenumber from your linear stability analysis
q_c = 0.53788  # Replace with your exact q_c


# ==========================================
# Core Simulator Function
# ==========================================
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / (dx ** 2)


def run_simulation(q_target, steps, noise_amp=0.0, record_every=10):
    U = np.ones((N, N)) * u_s
    V = np.ones((N, N)) * v_s

    x = np.arange(N) * dx
    X, _ = np.meshgrid(x, x)

    # Tiny perturbation for linear analysis
    perturbation = 0.0001 * np.cos(q_target * X) + noise_amp * np.random.randn(N, N)
    U += perturbation
    V += perturbation

    time_array = []
    amp_array = []

    for step in range(steps):
        uvv = U * V ** 2
        U += dt * (Du * laplacian(U) - uvv + F * (1 - U))
        V += dt * (Dv * laplacian(V) + uvv - (F + k) * V)

        if step % record_every == 0:
            V_fluct = V - np.mean(V)
            V_fft = np.fft.fft2(V_fluct) / (N * N)
            amp = np.max(np.abs(V_fft))
            time_array.append(step * dt)
            amp_array.append(amp)

    return np.array(time_array), np.array(amp_array), V


# ==========================================
# Analysis Functions
# ==========================================
def linear_growth_func(t, A0, mu_sigma):
    return A0 * np.exp(mu_sigma * t)




def extract_mu_sigma_and_g():
    print(f"--- Running Phase 1: Extracting mu*sigma and g at q_c = {q_c} ---")
    t, A, _ = run_simulation(q_target=q_c, steps=5000, noise_amp=0.0, record_every=50)

    # 1. Fit linear growth (use only early times before saturation)
    # Adjust the index '15' depending on how fast your pattern saturates
    t_early, A_early = t[:15], A[:15]
    popt, _ = curve_fit(linear_growth_func, t_early, A_early, p0=[A[0], 0.01])
    A0_fit, mu_sigma_fit = popt

    # 2. Extract saturation amplitude
    A_sat = np.mean(A[-10:])  # Average of last few points

    # 3. Calculate g
    g_fit = mu_sigma_fit / (A_sat ** 2)

    print(f"Calculated mu*sigma: {mu_sigma_fit:.6f}")
    print(f"Calculated Saturation Amplitude (A_sat): {A_sat:.6f}")
    print(f"Calculated g: {g_fit:.6f}\n")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(t, A, 'b.-', label="Numerical A(t)")
    plt.plot(t_early, linear_growth_func(t_early, *popt), 'r--', label=f"Linear Fit ($\mu\sigma$={mu_sigma_fit:.4f})")
    plt.axhline(A_sat, color='g', linestyle=':', label=f"Saturation ($A_{{sat}}$={A_sat:.4f})")
    plt.yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Amplitude (Log Scale)")
    plt.title("Linear Growth and Nonlinear Saturation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return mu_sigma_fit


def shifted_parabola_func(K, mu_sigma_fixed, xi_sq, K_shift):
    return mu_sigma_fixed - xi_sq * ((K - K_shift) ** 2)


def extract_coherence_length(mu_sigma_fixed):
    print("--- Running Phase 2: Extracting Coherence Length xi_0^2 (Shifted Fit) ---")
    K_vals = np.linspace(-0.04, 0.04, 9)
    lambdas = []

    for K in K_vals:
        q_test = q_c + K
        t, A, _ = run_simulation(q_target=q_test, steps=500, noise_amp=0.0, record_every=10)
        popt, _ = curve_fit(linear_growth_func, t, A, p0=[A[0], 0.01])
        lambdas.append(popt[1])

    lambdas = np.array(lambdas)

    # Fit the SHIFTED parabola to lambda(K)
    # p0 = [initial_guess_xi_sq, initial_guess_K_shift]
    popt_para, _ = curve_fit(
        lambda K, xi_sq, K_shift: shifted_parabola_func(K, mu_sigma_fixed, xi_sq, K_shift),
        K_vals, lambdas, p0=[1.0, -0.01]
    )
    xi_sq_fit, K_shift_fit = popt_para

    print(f"Calculated xi_0^2 (Curvature): {xi_sq_fit:.6f}")
    print(f"Calculated K_shift (Numerical Offset): {K_shift_fit:.6f}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(K_vals, lambdas, 'ko', label=r"Measured Growth Rates $\lambda(K)$")

    K_smooth = np.linspace(min(K_vals), max(K_vals), 100)
    plt.plot(K_smooth, shifted_parabola_func(K_smooth, mu_sigma_fixed, xi_sq_fit, K_shift_fit),
             'r-', label=rf"Shifted Fit ($\xi_0^2$={xi_sq_fit:.4f})")

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.axvline(K_shift_fit, color='blue', linestyle=':', label=rf"Peak Shift ($K_{{shift}}$={K_shift_fit:.4f})")
    plt.xlabel(r"$K = q - q_c$")
    plt.ylabel(r"Growth Rate $\lambda$")
    plt.title("Dispersion Relation (Shifted Parabola)")
    plt.legend()
    plt.grid(True)
    plt.show()


def simulate_eckhaus():
    print("--- Running Phase 3: Eckhaus Instability ---")
    # Force a wavenumber significantly outside the expected stable band
    # You may need to tune this offset (+0.03) based on your analytical Eckhaus boundary
    q_unstable = q_c + 0.03

    print(f"Targeting unstable wavenumber q = {q_unstable}")
    # Add noise so the perfect symmetry breaks, allowing the phase slip
    t, A, V_final = run_simulation(q_target=q_unstable, steps=15000, noise_amp=0.005, record_every=100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t, A, 'b-', linewidth=2)
    ax1.set_title("Amplitude vs Time (Watch for the dip!)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude |A|")
    ax1.grid(True)

    im = ax2.imshow(V_final, cmap='inferno', extent=[0, N * dx, 0, N * dx])
    ax2.set_title("Final Pattern (Look for Phase Slips/Defects)")
    fig.colorbar(im, ax=ax2)

    plt.show()


# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Get mu_sigma and g
    mu_sigma_measured = extract_mu_sigma_and_g()

    # 2. Use the measured mu_sigma to find xi_0^2
    extract_coherence_length(mu_sigma_measured)

    # 3. Observe the Eckhaus instability
    simulate_eckhaus()