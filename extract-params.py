import numpy as np
from scipy.optimize import fsolve


def find_critical_turing_point(k, Du, Dv, F_guess=0.053):
    """
    Finds the exact critical F_c and steady states for a given k.
    """

    def turing_boundary_condition(F):
        # 1. Calculate the steady state v_c for the guessed F
        discriminant = F ** 2 - 4 * F * (F + k) ** 2

        # If discriminant is negative, no steady state exists here
        if discriminant < 0:
            return 1e6  # Return a large penalty to guide the solver away

        v_c = (F + np.sqrt(discriminant)) / (2 * (F + k))

        # 2. Calculate Jacobian elements
        fu = -v_c ** 2 - F
        gv = F + k
        det_J = (F + k) * (v_c ** 2 - F)

        # 3. The Turing boundary condition:
        # (Dv*fu + Du*gv) - 2 * sqrt(Du * Dv * det_J) = 0

        # Safety check for negative square roots
        if det_J < 0 or (Dv * fu + Du * gv) < 0:
            return 1e6

        return (Dv * fu + Du * gv) - 2 * np.sqrt(Du * Dv * det_J)

    # Solve for F_c
    F_c_solution, info, ier, mesg = fsolve(turing_boundary_condition, F_guess, full_output=True)

    if ier != 1:
        raise ValueError(f"Solver failed to find a critical point: {mesg}")

    F_c = F_c_solution[0]

    # Calculate the exact u_c, v_c, and q_c at this F_c
    discriminant = F_c ** 2 - 4 * F_c * (F_c + k) ** 2
    v_c = (F_c + np.sqrt(discriminant)) / (2 * (F_c + k))
    u_c = (F_c + k) / v_c

    fu = -v_c ** 2 - F_c
    gv = F_c + k
    q_c = np.sqrt((Dv * fu + Du * gv) / (2 * Du * Dv))

    return F_c, u_c, v_c, q_c


# ==========================================
# Run the extraction for your parameters
# ==========================================
Du, Dv = 0.16, 0.08
k = 0.0616  # Your fixed simulation k

F_c, u_c, v_c, q_c = find_critical_turing_point(k, Du, Dv)

print(f"--- Analytical Critical Values at k = {k} ---")
print(f"F_c = {F_c:.6f}")
print(f"u_c = {u_c:.6f}")
print(f"v_c = {v_c:.6f}")
print(f"q_c = {q_c:.6f}")