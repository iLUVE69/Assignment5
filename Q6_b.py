import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Dynamics function for the SCARA robot
def scara_dynamics(t, q, tau):
    m1 = 1.0  
    m2 = 1.0
    l1 = 1.0  
    l2 = 1.0
    g = 9.81 

    q1, q2, q3, q1_dot, q2_dot, q3_dot = q  

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

# PID controller for joint control (Simple Independent Joint Control)
def pid_controller(error, prev_error, integral, Kp, Ki, Kd):
    proportional = Kp * error
    integral += Ki * error
    derivative = Kd * (error - prev_error)
    return proportional + integral + derivative, error

def control_loop(t, q, setpoint, prev_error, integral):
    Kp = 1.0 
    Ki = 0.1 
    Kd = 0.01

    # Calculate error
    error = setpoint - q[:3]

    # Calculate control 
    control_input = np.zeros(3)
    for i in range(3):
        control_input[i], prev_error[i] = pid_controller(error[i], prev_error[i], integral[i], Kp, Ki, Kd)

    return control_input, error, prev_error

# Initial conditions and time array
initial_conditions = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0])  # [q1, q2, q3, q1_dot, q2_dot, q3_dot]
time_span = (0, 10)  
time_steps = 1000  

# trajectory (a simple sinusoidal trajectory)
def desired_trajectory(t):
    return np.array([0.5 * np.sin(0.5 * t), 0.5 * np.cos(0.5 * t), 0.1 * np.sin(t)])

# Stochastic disturbance 
def stochastic_disturbance():
    return np.random.uniform(-0.01, 0.01, 3)

def simulate_scara_robot():
    control_output = []
    error_history = []
    prev_error = np.zeros(3)
    integral = np.zeros(3)

    def control_fun(t, q):
        setpoint = desired_trajectory(t)
        control_input, error, prev_error = control_loop(t, q, setpoint, prev_error, integral)
        control_output.append(control_input)
        error_history.append(error)
        return control_input

    sol = solve_ivp(scara_dynamics, time_span, initial_conditions,fun=control_fun, dense_output=True)

    return sol, control_output, error_history

# Perform simulation
solution, control_output, error_history = simulate_scara_robot()

# Plotting joint angles and velocities
time = solution.t
joint_angles = solution.y[:3].T
joint_velocities = solution.y[3:].T

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(time, joint_angles)
plt.xlabel('Time')
plt.ylabel('Joint Angles')
plt.legend(['q1', 'q2', 'q3'])
plt.title('Controlled Joint Angles over Time')

plt.subplot(212)
plt.plot(time, joint_velocities)
plt.xlabel('Time')
plt.ylabel('Joint Velocities')
plt.legend(['q1_dot', 'q2_dot', 'q3_dot'])
plt.title('Controlled Joint Velocities over Time')

plt.tight_layout()
plt.show()
