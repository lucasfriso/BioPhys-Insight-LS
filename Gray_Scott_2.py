import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==========================================
# 1. System Parameters & Discretization
# ==========================================
Du, Dv = 0.16, 0.08
F, k = 0.022, 0.051  # Parameters for Stripe patterns
dt = 1.0  # Time step
dx = 1.0  # Spatial step
N = 128  # Grid resolution (NxN)
total_steps = 10000  # Total integration steps
record_every = 100  # How often to compute the FFT/amplitude


# ==========================================
# 2. Initialization (Robust Seeding)
# ==========================================
# Start entirely in the stable trivial state
U = np.ones((N, N))
V = np.zeros((N, N))

# Create a strong, localized disturbance in the center to nucleate the pattern
r = 10  # radius of the perturbation
center = N // 2
U[center-r:center+r, center-r:center+r] = 0.5
V[center-r:center+r, center-r:center+r] = 0.25

# Add random noise everywhere to trigger secondary instabilities (like zigzag)
# as the pattern spreads outward
U += 0.05 * np.random.random((N, N))
V += 0.05 * np.random.random((N, N))

# ==========================================
# 3. Fourier Space Setup
# ==========================================
# Compute frequencies for the FFT
freqs = np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(freqs, freqs)
K_mag = np.sqrt(KX ** 2 + KY ** 2)

# Arrays to store time evolution data
time_array = []
amplitude_array = []


# ==========================================
# 4. Helper Functions
# ==========================================
def laplacian(Z):
    """5-point stencil Laplacian with periodic boundaries."""
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z) / (dx ** 2)


def extract_amplitude(V_field):
    """Extracts the amplitude of the dominant pattern using 2D FFT."""
    # Subtract the mean to remove the zero-frequency (homogeneous) mode
    V_fluct = V_field - np.mean(V_field)

    # Compute 2D FFT and normalize
    V_fft = np.fft.fft2(V_fluct) / (N * N)

    # Get the magnitude spectrum
    power_spectrum = np.abs(V_fft)

    # Return the maximum amplitude (the dominant mode)
    return np.max(power_spectrum), power_spectrum


# ==========================================
# 5. Main Simulation Loop
# ==========================================
print("Starting simulation...")
final_power_spectrum = None

for step in range(total_steps):
    # Reaction kinetics
    uvv = U * V ** 2

    # Forward Euler integration
    U += dt * (Du * laplacian(U) - uvv + F * (1 - U))
    V += dt * (Dv * laplacian(V) + uvv - (F + k) * V)

    # Record data periodically
    if step % record_every == 0:
        amp, power_spectrum = extract_amplitude(V)
        time_array.append(step * dt)
        amplitude_array.append(amp)
        final_power_spectrum = power_spectrum

        if step % 1000 == 0:
            print(f"Step {step}/{total_steps} | Dominant Amplitude: {amp:.5f}")

print("Simulation complete.")

# ==========================================
# 6. Visualization & Analysis
# ==========================================
fig = plt.figure(figsize=(15, 5))

# Plot 1: Real Space Pattern (V concentration)
ax1 = fig.add_subplot(131)
im1 = ax1.imshow(V, cmap='inferno', extent=[0, N*dx, 0, N*dx])
ax1.set_title("Final Pattern (Real Space)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Plot 2: Fourier Space (Power Spectrum)
shifted_spectrum = np.fft.fftshift(final_power_spectrum)
freqs_shifted = np.fft.fftshift(freqs)

# --- THE FIX: Safely calculate bounds for LogNorm ---
vmax_val = np.max(shifted_spectrum)
vmin_val = 1e-6
# If the pattern died out, force vmax to be slightly larger than vmin to prevent crashes
if vmax_val <= vmin_val:
    vmax_val = vmin_val * 10

ax2 = fig.add_subplot(132)
im2 = ax2.imshow(shifted_spectrum, cmap='viridis',
                 norm=LogNorm(vmin=vmin_val, vmax=vmax_val),
                 extent=[freqs_shifted[0], freqs_shifted[-1], freqs_shifted[0], freqs_shifted[-1]])
ax2.set_title("2D Power Spectrum (Fourier Space)")
ax2.set_xlabel(r"$k_x$")
ax2.set_ylabel(r"$k_y$")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Plot 3: Amplitude Evolution over Time
ax3 = fig.add_subplot(133)
ax3.plot(time_array, amplitude_array, 'b-', linewidth=2)
ax3.set_title("Dominant Mode Amplitude $A(t)$")
ax3.set_xlabel("Time")
ax3.set_ylabel("Amplitude $|A|$")
ax3.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()