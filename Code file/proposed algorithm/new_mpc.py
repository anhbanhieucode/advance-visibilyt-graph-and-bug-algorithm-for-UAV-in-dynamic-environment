import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def my_mpc(path, T=5):  # Set default prediction horizon to 5
    # UAV parameters
    m = 0.698  # mass (adjust as needed)
    g = 9.81  # gravity
    Ix, Iy, Iz = 0.0034, 0.0034, 0.006  # moments of inertia
    Fd_u, Fd_v, Fd_w = 0.1, 0.1, 0.1  # drag terms
    Jtp = 1.302 * 10**(-6)  # N*m*s^2=kg*m^2
    ct = 7.6184 * 10**(-8) * (60 / (2 * np.pi)) ** 2  # N*s^2
    cq = 2.6839 * 10**(-9) * (60 / (2 * np.pi)) ** 2  # N*m*s^2
    l = 0.171  # m

    dt = 0.05  # time step

    # Control input constraints
    u_min = np.array([-10.0, -10.0, -10.0, -10.0])  # Replace with actual limits
    u_max = np.array([20.5, 20.5, 20.5, 20.5])      # Replace with actual limits

    delta_U_min = ca.MX(np.array([-0.6, -0.5, -0.5, -0.5]))  # Replace with desired min deltas
    delta_U_max = ca.MX(np.array([0.5, 0.5, 0.5, 0.5]))      # Replace with desired max deltas

    # Define CasADi variables
    x = ca.MX.sym('x', 12)  # 12 states
    u = ca.MX.sym('u', 4)   # 4 control inputs

    # UAV nonlinear dynamics model
    u_dot = (x[1] * x[5] - x[2] * x[4]) + g * ca.sin(x[10]) - Fd_u / m
    v_dot = (x[2] * x[3] - x[0] * x[5]) - g * ca.cos(x[10]) * ca.sin(x[9]) - Fd_v / m
    w_dot = (x[0] * x[4] - x[1] * x[3]) - g * ca.cos(x[10]) * ca.cos(x[9]) + u[0] / m - Fd_w / m
    p_dot = x[4] * x[5] * (Iy - Iz) / Ix + u[1] / Ix
    q_dot = x[3] * x[5] * (Iz - Ix) / Iy + u[2] / Iy
    r_dot = x[3] * x[4] * (Ix - Iy) / Iz + u[3] / Iz

    # Define the full state derivative
    xdot = ca.vertcat(u_dot, v_dot, w_dot, p_dot, q_dot, r_dot, x[0], x[1], x[2], x[3], x[4], x[5])

    # Discretize the system using Euler integration
    x_next = x + dt * xdot
    f = ca.Function('f', [x, u], [x_next])

    # Define cost function for tracking target 3D positions
    x_targets = path
    x_target_vectors = np.zeros((12, T + 1))  # Each column is a target state for each time step
    x_target_vectors[6, :T] = x_targets[:, 0]  # x target
    x_target_vectors[7, :T] = x_targets[:, 1]  # y target
    print("x_target: ", x_target_vectors)
    # Initialize optimization variables
    X = ca.MX.sym('X', 12, T + 1)
    U = ca.MX.sym('U', 4, T)
    U_prev = ca.MX.zeros(4)  # Initial previous control input, assumed to be zero
    cost = 0
    g = []  # equality constraints for dynamics

    Q = np.eye(12)  # State cost matrix
    R = 1.5 * np.eye(4)  # Control cost matrix

    # Build the MPC problem
    for t in range(T):
        # State tracking cost
        state_deviation = X[:, t] - x_target_vectors[:, t]
        cost += ca.mtimes(state_deviation.T, Q @ state_deviation)  # State cost

        # Control effort cost
        if t == 0:
            delta_U = U[:, t] - U_prev
        else:
            delta_U = U[:, t] - U[:, t - 1]
        cost += ca.mtimes(delta_U.T, R @ delta_U)  # Control input change cost

        # Update dynamics constraints
        x_next = f(X[:, t], U[:, t])
        g.append(X[:, t + 1] - x_next)

    # Solve the optimization problem
    opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
    nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Bounds
    lbx = -np.inf * np.ones(opt_variables.shape[0])
    ubx = np.inf * np.ones(opt_variables.shape[0])
    total_constraints = sum(expr.shape[0] for expr in g)
    lbg = np.zeros(total_constraints)
    ubg = np.zeros(total_constraints)

# Compute the total size of state and control variables
    state_size = 12 * (T + 1)  # Total states (12 per time step for T+1 steps)
    control_start_index = state_size  # Control inputs start after the states
    control_size = 4 * T  # Total control inputs (4 per time step for T steps)

    # Set specific bounds for control inputs
    for t in range(T):
        lbx[control_start_index + t * 4:control_start_index + (t + 1) * 4] = u_min
        ubx[control_start_index + t * 4:control_start_index + (t + 1) * 4] = u_max


        # Solve
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    x_opt = sol['x']

    # Optimized control inputs and states
    the_U_opt = x_opt[-4 * T:].reshape((4, T))
    X_opt = x_opt[:12 * (T + 1)].reshape((12, T + 1))
    print("the_U_opt: ",the_U_opt)
    u_opt = the_U_opt[:, 0].toarray()
    print("Set of U: ", u_opt)
    print(type(u_opt))
    print("State: ", X_opt)
    # Mean of absolute values of the first control input set
    mean_abs_first_set = np.mean(np.abs(u_opt))
    print(mean_abs_first_set)
    xyz_opt = X_opt[6:9, :]
    print("xyz_opt: ", xyz_opt)
    x_coords = xyz_opt[0, :]
    y_coords = xyz_opt[1, :]

    return x_coords, y_coords, mean_abs_first_set, u_opt

if __name__ == "__main__":
    path = np.array([[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]])[:5]  # Provide path for 5 steps
    x_coords, y_coords, mean_abs_first_set, U = my_mpc(path)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(path[:, 0],path[:, 1],marker = 'o', linestyle= '-', color = 'red', label = "Path")
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', label='Optimized Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Optimized 2D Trajectory of UAV')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
