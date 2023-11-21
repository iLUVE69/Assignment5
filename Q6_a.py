import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def scara_dynamics(t, q, tau):
    # Constants and parameters
    m1 = 1.0  # Mass or inertial parameters for each link
    m2 = 1.0
    l1 = 1.0  # Length of links
    l2 = 1.0
    g = 9.81  # Gravity

    q1, q2, q3, q1_dot, q2_dot, q3_dot = q  # Joint positions and velocities

    # Mass matrix (M(q))
    M = np.array([
        [m1 * l1**2 + m2 * (l1**2 + l2**2 + 2 * l1 * l2 * np.cos(q2)), m2 * (l2**2 + l1 * l2 * np.cos(q2))],
        [m2 * (l2**2 + l1 * l2 * np.cos(q2)), m2 * l2**2]
    ])

    # Coriolis and centrifugal forces (C(q, q_dot))
    C = np.array([
        [-m2 * l1 * l2 * (2 * q3_dot * np.sin(q2) + q2_dot) * q3_dot * np.sin(q2)],
        [m2 * l1 * l2 * q2_dot * q3_dot * np.sin(q2)]
    ])

    # Gravitational forces (G(q))
    G = np.array([
        [m1 * g * l1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + l2 * np.cos(q1 + q2))],
        [m2 * g * l2 * np.cos(q1 + q2)]
    ])

    # Equations of motion (Euler-Lagrange equation)
    q_ddot = np.linalg.inv(M) @ (tau - C - G)

    return np.array([q1_dot, q2_dot, q3_dot, q_ddot[0][0], q_ddot[1][0], 0])  # Assuming q3_ddot is 0

# Initial conditions and time array
initial_conditions = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0])  # [q1, q2, q3, q1_dot, q2_dot, q3_dot]
time_span = (0, 10)  # Time range for simulation

tau_values = np.array([3.5, 3.5])  # [tau1, tau2]

solution = solve_ivp(scara_dynamics, time_span, initial_conditions, args=(tau_values,), dense_output=True)

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(solution.t, solution.y[:3].T)
plt.xlabel('Time')
plt.ylabel('Joint Angles')
plt.legend(['q1', 'q2', 'q3'])

plt.subplot(212)
plt.plot(solution.t, solution.y[3:].T)
plt.xlabel('Time')
plt.ylabel('Joint Velocities')
plt.legend(['q1_dot', 'q2_dot', 'q3_dot'])

plt.tight_layout()
plt.show()
