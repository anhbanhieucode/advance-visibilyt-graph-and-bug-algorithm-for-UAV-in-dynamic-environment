import numpy as np
import matplotlib.pyplot as plt
# Constants
m = 0.698  # mass in kg
g = 9.81   # gravity in m/s^2
Fd_u, Fd_v, Fd_w = 0.1, 0.1, 0.1  # drag terms
Ix, Iy, Iz = 0.0034, 0.0034, 0.006  # moments of inertia
# Initialize lists to store control inputs over time
l = 0.171 #m
U1_list, U2_list, U3_list, U4_list = [], [], [], []
# State initialization
x = np.zeros(12)  # Initial state (all zeros)
# Initialize lists to store position, velocity, and acceleration over time
position_x, position_y, position_z = [], [], []
velocity_u, velocity_v, velocity_w = [], [], []
acceleration_u, acceleration_v, acceleration_w = [], [], []

global T
T = 80  # Number of time steps

d = 2.5 #m

psi = 0/180*np.pi #radian

d_x = d*np.sin(psi) 
d_y = d*np.cos(psi) 
d_z = 0 

cal_t = 6 #second
t1 = 0.1
t2 = cal_t*3/8

dt = cal_t/T

x[0] = 0.0  # u (velocity in x)
# Inverse dynamics function to calculate control inputs
def inverse_dynamics(x, u_dot, v_dot, w_dot, r_dot):
    u, v, w, p, q, r, x0, y0, z0, phi, theta, psi = x
     # Calculate pitch angle (theta) using u_dot equation
    theta = np.arcsin((u_dot - v*r + w*q) / g)
    #print(theta)
     # Calculate roll angle (phi) using v_dot equation
    phi = np.arcsin(-(v_dot - w*p + u*r) / (g * np.cos(theta)))

    p  = (phi-x[9])/dt
    p_dot = (p-x[3])/dt
    q = (theta-x[10])/dt
    q_dot = (q-x[4])/dt

    dx[3] = p_dot
    dx[4] = q_dot
    dx[5] = w_dot
    dx[9]  = p
    dx[10] = q
    dx[11] = w

    # Calculate control inputs (U1, U2, U3, U4) using the inverse dynamics equations
    U1 = m * (w_dot + g * np.cos(theta) * np.cos(phi) - (u * q- v * p)) + 0.1
    U2 = Ix * (p_dot - (r * q * (Iy - Iz)/Ix) )
    U3 = Iy * (q_dot - (r * p * (Iz - Ix)/Iy) )
    U4 = Iz * (r_dot - (q * p * (Ix - Iy)/Iz) )
    
    return np.array([U1, U2, U3, U4])

def trapezoidal_velocity_profile(d, t, t_acc, T):
    
    a = d / (t_acc**2 + t_acc*(t - 2*t_acc))  
    
    # Create time array from 0 to total time
    time = np.linspace(0, t, int(T))  # Time vector (0 to total time)
    
    # Initialize the acceleration list
    acceleration = []
    
    for t_i in time:
        if t_i < t_acc:
            # Acceleration phase: constant acceleration
            acceleration.append(a)
        elif t_i < (t - t_acc):
            # Constant velocity phase: acceleration is zero
            acceleration.append(0)
        else:
            # Deceleration phase: constant deceleration
            acceleration.append(-a)
    
    return acceleration


#x_direction:
u_dot = trapezoidal_velocity_profile(d_x ,cal_t, t1 , T)
#print(u_dot)

#y_direction:
v_dot = trapezoidal_velocity_profile(d_y , cal_t, t2 , T)
#print(v_dot)

w_dot = np.zeros(T)
r_dot = np.zeros(T)
set_of_states = list(zip(u_dot,v_dot,w_dot,r_dot))
for state in set_of_states:
    print(state)


x9_list = []   # Pitch angle (x[9])
x3_list = []   # Angular velocity about x-axis (x[3])
dx3_list = []  # Rate of change of angular velocity (dx[3])

# Simulate for T time steps
t = 1
global dx
dx = np.zeros(12)  # Create a vector for state derivatives

for state in set_of_states:
    # Example derivatives (u_dot, v_dot, etc.) for this time step (you would normally provide this input)
    
    u_dot, v_dot, w_dot, r_dot = state
    # Safeguard against division by zero
    if x[7] != 0:
        psi = np.arctan(x[6] / x[7])
    else:
        psi = np.pi / 2 if x[6] > 0 else -np.pi / 2  # Assign 90° or -90° based on the sign of x[6]
    #print(psi)
    # Example control inputs (u) for this time step
    # Calculate control inputs using the inverse dynamics function
    U = inverse_dynamics(x, u_dot, v_dot, w_dot, r_dot)

    U1_list.append(U[0])
    U2_list.append(U[1])
    U3_list.append(U[2])
    U4_list.append(U[3])
    
    # Print the control inputs
    print(f"Control Inputs at t={t}: {U}")

    # Update the state `x` based on the control inputs (simple Euler integration)
    # The state vector x consists of [u, v, w, p, q, r, x0, y0, z0, phi, theta, psi]
    #dx = np.zeros(12)  # Create a vector for state derivatives

    # State derivatives (from the inverse dynamics equations)
    dx[0] = u_dot
    dx[1] = v_dot
    dx[2] = w_dot
    dx[6] = dx[6] + dx[0] * dt
    dx[7] = dx[7] + dx[1] * dt 
    dx[8] = dx[8] + dx[2] * dt 
    
     # Store values for plotting
    x9_list.append(x[9])   # Pitch angle
    x3_list.append(x[3])   # Angular velocity
    dx3_list.append(dx[3]) # Rate of change of angular velocity


  # Store values for position, velocity, and acceleration for plotting
    position_x.append(x[6])  # x0 (position in x)
    position_y.append(x[7])  # y0 (position in y)
    position_z.append(x[8])  # z0 (position in z)
    
    velocity_u.append(x[0])  # u (velocity in x)
    velocity_v.append(x[1])  # v (velocity in y)
    velocity_w.append(x[2])  # w (velocity in z)
    
    acceleration_u.append(u_dot)  # u_dot (acceleration in x)
    acceleration_v.append(v_dot)  # v_dot (acceleration in y)
    acceleration_w.append(w_dot)  # w_dot (acceleration in z)

    # Update state using Euler method (basic time-stepping integration)
    x = x + dx * dt  # Use time step `dt` to integrate the state

    # Ensure the state stays within reasonable bounds (if needed)
    # Example: You can use np.clip to constrain state variables if necessary
    # x = np.clip(x, -np.inf, np.inf)  # Example clipping

    # Optionally print the updated state to observe changes
    #print(f"Updated State at t={t}: {x[6],x[7],x[8]}")
    #print(f"Updated State at t={t}: {x[9],x[10],x[11]}")
    t +=  1

# Plotting the control inputs
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), U1_list, label='U1', color='r')
plt.plot(range(1, T + 1), U2_list, label='U2', color='g')
plt.plot(range(1, T + 1), U3_list, label='U3', color='b')
plt.plot(range(1, T + 1), U4_list, label='U4', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Control Inputs')
plt.title('Control Inputs (U1, U2, U3, U4) Over Time')
plt.legend()
plt.grid()
plt.show()

'''
# Plot x[9] (pitch angle), x[3] (angular velocity), and dx[3] (angular acceleration)
plt.figure(figsize=(10, 6))
plt.plot(range(1, T + 1), x9_list, label='x[9] (Pitch angle)', color='r')
plt.plot(range(1, T + 1), x3_list, label='x[3] (Angular velocity)', color='g')
plt.plot(range(1, T + 1), dx3_list, label='dx[3] (Angular acceleration)', color='b')
plt.xlabel('Time Step')
plt.ylabel('Values')
plt.title('Pitch Angle, Angular Velocity, and Angular Acceleration Over Time')
plt.legend()
plt.grid()
plt.show()
'''

plt.figure(figsize=(12, 8))
# Plot Position
plt.subplot(3, 1, 1)
plt.plot(range(1, T + 1), position_x, label='Position in X (x0)', color='r')
plt.plot(range(1, T + 1), position_y, label='Position in Y (y0)', color='g')
plt.plot(range(1, T + 1), position_z, label='Position in Z (z0)', color='b')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.title('Position Over Time')
plt.legend()
plt.grid()

# Plot Velocity
plt.subplot(3, 1, 2)
plt.plot(range(1, T + 1), velocity_u, label='Velocity in X (u)', color='r')
plt.plot(range(1, T + 1), velocity_v, label='Velocity in Y (v)', color='g')
plt.plot(range(1, T + 1), velocity_w, label='Velocity in Z (w)', color='b')
plt.xlabel('Time Step')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Over Time')
plt.legend()
plt.grid()

# Plot Acceleration
plt.subplot(3, 1, 3)
plt.plot(range(1, T + 1), acceleration_u, label='Acceleration in X (u_dot)', color='r')
plt.plot(range(1, T + 1), acceleration_v, label='Acceleration in Y (v_dot)', color='g')
plt.plot(range(1, T + 1), acceleration_w, label='Acceleration in Z (w_dot)', color='b')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
