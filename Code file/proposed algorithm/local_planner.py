import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
from new_mpc import my_mpc
import time
# Define the start and goal points
start_point = np.array([1, 1])  # Starting position (x, y)
goal_point = np.array([9, 9])   # Goal position (x, y)
real_goal = goal_point.copy()   # Store the real goal
total_time = 0

preset_purple_circles_10 = [
    [1.2, 7.8], [8.3, 2.4], [3.5, 5.7], [9.1, 3.9], [2.8, 9.2],
    [5.4, 3.8], [7.9, 8.4], [6.1, 2.5], [4.7, 9.6], [1.0, 6.3]
]

preset_purple_circles_50 = [
    [2.5, 9.7], [7.8, 3.2], [4.1, 5.6], [9.3, 2.9], [0.7, 6.1],
    [3.6, 8.5], [5.9, 4.7], [1.4, 2.0], [8.2, 6.4], [9.0, 1.1],
    [6.8, 8.2], [7.1, 3.5], [4.9, 9.4], [6.3, 1.7], [2.1, 7.0],
    [0.3, 8.3], [8.0, 4.9], [1.9, 2.2], [9.6, 5.1], [3.0, 6.6],
    [4.3, 1.8], [7.6, 9.3], [6.7, 2.6], [0.9, 7.4], [8.7, 3.0],
    [3.8, 9.9], [7.5, 5.3], [5.5, 6.8], [1.2, 4.0], [9.4, 2.7],
    [0.6, 3.9], [6.9, 0.8], [2.4, 6.9], [8.9, 5.8], [3.7, 1.2],
    [7.2, 4.2], [1.3, 5.0], [4.0, 2.3], [9.8, 6.0], [5.2, 3.4],
    [8.4, 9.5], [0.2, 4.4], [3.9, 7.3], [6.2, 1.9], [9.7, 8.1],
    [1.8, 2.8], [7.4, 6.5], [4.6, 8.1], [2.3, 9.1], [5.8, 4.5],
    [6.0, 7.9], [9.2, 3.3], [3.3, 0.5], [7.3, 1.5], [2.7, 7.2]
]

preset_purple_circles_100 = [
    [1.4, 2.7], [4.1, 9.2], [6.5, 1.9], [2.6, 4.4], [3.9, 8.7],
    [0.8, 6.0], [7.2, 5.3], [6.3, 9.6], [4.7, 1.5], [1.0, 2.3],
    [8.1, 7.5], [3.0, 9.3], [7.4, 4.0], [6.9, 3.1], [5.2, 8.9],
    [8.3, 1.0], [9.8, 2.9], [1.2, 6.2], [9.6, 5.1], [0.7, 8.4],
    [2.0, 1.3], [6.4, 4.9], [5.0, 7.6], [9.3, 3.8], [7.9, 0.2],
    [8.5, 6.8], [3.3, 0.9], [6.6, 2.5], [9.7, 9.9], [1.9, 8.2],
    [7.3, 4.6], [2.2, 3.8], [5.4, 2.0], [0.5, 9.0], [8.0, 5.6],
    [9.4, 7.1], [6.8, 1.8], [4.0, 7.8], [3.7, 4.2], [2.4, 6.6],
    [8.9, 8.3], [7.7, 0.3], [5.5, 9.1], [6.2, 5.7], [3.8, 0.7],
    [0.3, 4.8], [9.2, 2.1], [4.5, 7.2], [7.0, 3.6], [8.6, 1.2],
    [2.5, 9.8], [1.3, 5.5], [3.6, 2.9], [5.1, 0.4], [6.0, 6.7],
    [2.9, 3.5], [9.5, 8.0], [7.1, 7.7], [5.9, 4.3], [8.4, 3.2],
    [4.4, 1.7], [0.9, 6.3], [7.8, 8.8], [2.1, 4.5], [1.6, 2.8],
    [5.3, 0.8], [4.2, 5.0], [9.0, 1.4], [2.7, 0.6], [6.7, 9.0],
    [8.2, 4.3], [7.6, 2.3], [1.7, 3.4], [0.6, 9.7], [3.5, 9.1],
    [6.1, 2.1], [4.3, 9.5], [1.5, 8.3], [7.3, 7.0], [3.2, 5.3],
    [5.7, 9.4], [9.1, 6.5], [4.6, 7.9], [6.0, 3.7], [1.8, 7.4],
    [0.4, 1.1], [5.8, 3.0], [9.9, 5.8], [3.1, 0.1], [4.9, 6.9],
    [7.0, 2.8], [2.8, 0.2], [9.4, 3.0], [0.2, 2.4], [8.7, 9.3],
    [5.0, 8.2], [1.8, 4.1], [3.4, 7.3], [6.3, 1.4], [9.5, 2.7]
]


# Parameters
m = 0.698  # mass (adjust as needed)
g = 9.81  # gravity
Ix, Iy, Iz = 0.0034, 0.0034, 0.006  # moments of inertia
Fd_u, Fd_v, Fd_w = 0.1, 0.1, 0.1  # drag terms
Jtp = 1.302 * 10**(-6)  # N*m*s^2=kg*m^2
ct = 7.6184 * 10**(-8) * (60 / (2 * np.pi)) ** 2  # N*s^2
cq = 2.6839 * 10**(-9) * (60 / (2 * np.pi)) ** 2  # N*m*s^2
l = 0.171  # m

global state
# Initial state [u, v, w, p, q, r, x, y, z, roll, pitch, yaw]
state = np.array([0, 0, 0, 0, 0, 0, start_point[0], start_point[1], 1, 0, 0, 0], dtype=np.float64)


def uav_dynamics(U, m, g, Ix, Iy, Iz):
    # Velocity dynamics
    u_dot = (state[1] * state[5] - state[2] * state[4]) + g * np.sin(state[10]) - Fd_u * state[0] / m
    v_dot = (state[2] * state[3] - state[0] * state[5]) - g * np.cos(state[10]) * np.sin(state[9]) - Fd_v * state[1] / m
    w_dot = (state[0] * state[4] - state[1] * state[3]) - g * np.cos(state[10]) * np.cos(state[9]) + U[0] / m - Fd_w * state[2] / m

    # Angular dynamics
    p_dot = state[4] * state[5] * (Iy - Iz) / Ix + U[1] / Ix
    q_dot = state[3] * state[5] * (Iz - Ix) / Iy + U[2] / Iy
    r_dot = state[3] * state[4] * (Ix - Iy) / Iz + U[3] / Iz

    result = np.array([
            float(u_dot), 
            float(v_dot), 
            float(w_dot), 
            float(p_dot), 
            float(q_dot), 
            float(r_dot)
        ], dtype=np.float64)

    # Debug output
    print("u_dot:", u_dot, "v_dot:", v_dot, "w_dot:", w_dot)
    print("p_dot:", p_dot, "q_dot:", q_dot, "r_dot:", r_dot)
    print(result)
    print("Result shape:", len(result))
    # Return the derivatives of the state
    return result

def state_update(U,dt):
    # Sample control input from U
    U_sample = U
    print(type(U))
    # Calculate dynamics and update state
    state_dot = uav_dynamics(U_sample, m, g, Ix, Iy, Iz)

    # Update velocities based on dynamics
    state[0] += state_dot[0] * dt  # Update u
    state[1] += state_dot[1] * dt  # Update u
    state[2] += state_dot[2] * dt  # Update u
    state[3] += state_dot[3] * dt  # Update u
    state[4] += state_dot[4] * dt  # Update u
    state[5] += state_dot[5] * dt  # Update u
 

    # Update positions based on velocities
    state[6] += state[0] * dt  # Update x position
    state[7] += state[1] * dt  # Update y position
    state[8] += state[2] * dt  # Update z position (z is not displayed in 2D)
    state[9] += state[3] * dt  # Update roll (not displayed)
    state[10] += state[4] * dt  # Update pitch (not displayed)
    state[11] += state[5] * dt  # Update yaw (not displayed)

    # Update UAV position for animation
    return state[6], state[7]  # Update position based on x[6], x[7]

# Circle properties
circle_radius = 2.0  # Radius of the yellow circle
purple_radius = 0.4  # Radius of the purple circles
num_circle_points = 50  # Number of points composing the yellow circle

# Generate angles for the circle points
angles = np.linspace(0, 2 * np.pi, num_circle_points)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.scatter(start_point[0], start_point[1], color="red", s=50, label="Start Point")
ax.scatter(goal_point[0], goal_point[1], color="green", s=50, label="Goal Point")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.legend()

# Create the animated objects
circle, = plt.plot([], [], 'bo', markersize=10)
yellow_circle_points, = plt.plot([], [], 'yo', markersize=5)
chosen_point, = plt.plot([], [], 'o', color='orange', markersize=8)

# Plot for displaying goals
goal_points_plot, = plt.plot([], [], 'ro', markersize=5, label="Goals")
ax.legend()

# List to store purple circles added by mouse click
user_circles = []

# Initialize current position of the blue point
current_position = start_point.astype(float)
move_speed = 4  # Speed factor for each frame
threshold = 15  # Angle threshold

# Global variables
current_goal_index = 0
goals = []  # Initialize goals to avoid unbound reference


def interpolate_2d_points(original_points, m):
    """
    Interpolate original 2D points (x, y) to create m points.

    Parameters:
    - original_points: 2D array-like (original data points as [[x1, y1], [x2, y2], ...])
    - m: int (number of points to generate)

    Returns:
    - interpolated_points: 2D numpy array (m interpolated points as [[x1, y1], [x2, y2], ...])
    """
    # Ensure original points are a numpy array
    original_points = np.asarray(original_points)
    
    # Split original points into x and y components
    x_original = original_points[:, 0]  # x-coordinates
    y_original = original_points[:, 1]  # y-coordinates
    
    # Create normalized x-coordinates for the original points
    n = len(x_original)
    x_normalized = np.linspace(0, 1, n)

    # Create normalized x-coordinates for the new m points
    x_interpolated = np.linspace(0, 1, m)

    # Create interpolation functions for x and y
    interp_func_x = interp1d(x_normalized, x_original, kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(x_normalized, y_original, kind='linear', fill_value='extrapolate')

    # Generate the interpolated points
    x_new = interp_func_x(x_interpolated)
    y_new = interp_func_y(x_interpolated)

    # Combine interpolated x and y into a 2D array
    interpolated_points = np.vstack((x_new, y_new)).T

    return interpolated_points


# Function to find points sorted by angle to the goal
def find_sorted_angle_points(point_set, current_point, goal_point):
    vector_to_goal = (goal_point - current_point).astype(float)
    vector_to_goal /= np.linalg.norm(vector_to_goal)
    angle_point_list = [
        (np.arccos(np.clip(np.dot(vector_to_goal, (point - current_point) / np.linalg.norm(point - current_point)), -1.0, 1.0)), point)
        for point in point_set if np.linalg.norm(point - current_point) > 0
    ]
    angle_point_list.sort(key=lambda x: x[0])
    sorted_points = [point for _, point in angle_point_list]
    return sorted_points, angle_point_list

# Initialization function to reset positions
def init():
    circle.set_data([], [])
    yellow_circle_points.set_data([], [])
    chosen_point.set_data([], [])
    goal_points_plot.set_data([], [])
    return circle, yellow_circle_points, chosen_point, goal_points_plot

def interpolate_three_points(p1, p2, p3, num_points=50):
    points_per_segment = num_points // 2
    if points_per_segment < 2:
        points_per_segment = 2

    path_1_to_2 = np.linspace(p1, p2, points_per_segment)
    path_2_to_3 = np.linspace(p2, p3, num_points - points_per_segment)

    full_path = np.concatenate((path_1_to_2, path_2_to_3), axis=0)
    return full_path


cal_time = []
def animate(i):
    global current_position, goal_point, real_goal, goals, current_goal_index, previous_pos, total_time, state

    lidar_data = []
    x_points = current_position[0] + circle_radius * np.cos(angles)
    y_points = current_position[1] + circle_radius * np.sin(angles)

    valid_x_points, valid_y_points = [], []
    for x, y in zip(x_points, y_points):
        exclude_yellow_point = False
        blue_to_yellow_distance = np.linalg.norm(np.array([x, y]) - current_position)

        for purple_circle in user_circles:
            px, py = purple_circle.get_data()


            purple_center = np.array([px[0], py[0]])
            blue_to_purple_distance = np.linalg.norm(purple_center - current_position)

            if blue_to_purple_distance > blue_to_yellow_distance:
                continue
            blue_to_yellow = np.array([x, y]) - current_position
            blue_to_purple = purple_center - current_position

            if np.dot(blue_to_yellow, blue_to_purple) > 0:
                projection_length = np.dot(blue_to_purple, blue_to_yellow) / np.dot(blue_to_yellow, blue_to_yellow)
                projection_point = current_position + projection_length * blue_to_yellow
                if np.linalg.norm(projection_point - purple_center) < purple_radius:
                    exclude_yellow_point = True
                    break

        if not exclude_yellow_point:
            valid_x_points.append(x)
            valid_y_points.append(y)
            lidar_data.append([x, y])

    yellow_circle_points.set_data(valid_x_points, valid_y_points)

    if lidar_data:
        lidar_data = np.array(lidar_data)
        points, angles_list = find_sorted_angle_points(lidar_data, current_position, goal_point)
        angle_1 = np.degrees(angles_list[0][0])
        angle_2 = np.degrees(angles_list[1][0])

        if angle_1 > 38 and angle_2 > 38 and abs(angle_1 - angle_2) < threshold:
            if np.linalg.norm(current_position - goal_point) < 1 :
                goal_point = real_goal
            else:
                goal_point = points[0]
                goals = interpolate_three_points(current_position, points[0], real_goal)
                current_goal_index = 0  # Reset goal index for new path

        # Update goal point to next point in the list if it's reached
        if len(goals) > 0 and current_goal_index < len(goals):
            goal_point = goals[current_goal_index]
            if np.linalg.norm(current_position - goal_point) < 1:  # When close to goal point
                current_goal_index += 1  # Move to next goal

        print("real_goal", real_goal)
        print("goal_point", goal_point)

        # Update goal points plot
        goal_points_plot.set_data(goals[:, 0], goals[:, 1]) if len(goals) > 0 else goal_points_plot.set_data([], [])
        start = time.time()
        # Choose the next point towards the goal
        state_2d_targets = [
            current_position + (points[0] - current_position) / 2,
            current_position + (points[1] - current_position) / 2
        ]
        #print("current_pos: ", current_position)
        #print("points: ", state_2d_targets)
        mean_U_list = []
        for points in state_2d_targets:
            T = 4
            inter_point = interpolate_2d_points([current_position , points],T)
            #print(inter_point)
            x_tra,y_tra, mean_U, U  = my_mpc(inter_point,T)
            #U, mean_U  = my_mpc(inter_point,T,x)
            #ax.scatter(x_tra, y_tra, color="yellow", s=50, label="predict positions")
            mean_U_list.append((points,U,mean_U)) 
            print(x_tra,y_tra)
        mean_U_list.sort(key = lambda item: item[2] )

        #x_current = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, current_position[0], current_position[1], 0])  # Initial state
        #chosen_position, _ = left_or_right(state_2d_targets, x_current, move_speed)
        best_target, chosen_U ,best_mean_U = mean_U_list[0]
        #print("chosen_U: ",chosen_U)
        #print("state: ",state)
        #arget_x, target_y = state_update(chosen_U,3)
        chosen_point.set_data(best_target[0], best_target[1])
        #chosen_point.set_data(target_x, target_y)
        end = time.time()
        elapsed_time = end-start
        cal_time.append(elapsed_time)
        total_time += elapsed_time
        print("time: ",elapsed_time)
        print('Total time: ',total_time)

    if chosen_point.get_data()[0]:  # Check if the chosen point exists
        chosen_point_pos = np.array(chosen_point.get_data())
        direction = (chosen_point_pos - current_position) / np.linalg.norm(chosen_point_pos - current_position)
        previous_pos = current_position
        current_position += direction * 0.01 * move_speed
        #current_position = [target_x,target_y]

    circle.set_data(current_position[0], current_position[1])
    #circle.set_data(target_x, target_y)


    return [circle, yellow_circle_points, chosen_point, goal_points_plot] + user_circles



for pos in preset_purple_circles_10:
    purple_circle, = ax.plot(pos[0], pos[1], 'o', color = 'purple', markersize=20)  # Create purple circles at predefined positions
    #purple_circle.set_data(pos[0], pos[1])
    user_circles.append(purple_circle)

# Mouse click event handler to add a purple circle at the clicked position
def on_click(event):
    if not event.inaxes:
        return
    new_circle, = ax.plot(event.xdata, event.ydata, 'o', color='purple', markersize=15)
    user_circles.append(new_circle)

# Connect the click event to the on_click function
fig.canvas.mpl_connect('button_press_event', on_click)

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=200, interval=50, blit=True)

# Display the animation
plt.show()
cal_time_sorted = sorted(cal_time)

# Print each element on a new line
print("Sorted cal_time:")
for time in cal_time_sorted:
    print(time)