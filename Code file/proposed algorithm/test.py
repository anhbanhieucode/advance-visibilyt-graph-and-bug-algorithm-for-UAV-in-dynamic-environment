import matplotlib.pyplot as plt
import numpy as np

def direction_to_goal(current, goal):
    dx, dy = goal[0] - current[0], goal[1] - current[1]
    magnitude = (dx**2 + dy**2) ** 0.5
    return (0, 0) if magnitude == 0 else (round(dx / magnitude), round(dy / magnitude))

def is_obstacle(point, obstacles):
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1), (x2, y2) in obstacles)

def follow_boundary(current, obstacles, last_move):
    for (x1, y1), (x2, y2) in obstacles:
        x, y = current
        if last_move == "left" or (x == x1 and y < y2): return (x, y + 1), "up"
        if last_move == "up" or (y == y2 and x < x2): return (x + 1, y), "right"
        if last_move == "right" or (x == x2 and y > y1): return (x, y - 1), "down"
        if last_move == "down" or (y == y1 and x > x1): return (x - 1, y), "left"
    return current, last_move

def can_leave_to_goal(current, goal, obstacles):
    x, y = current
    gx, gy = goal
    for (x1, y1), (x2, y2) in obstacles:
        if (x1 <= x <= x2 and y1 <= y <= y2) or (x1 <= gx <= x2 and y1 <= gy <= y2):
            return False
    return True

def bug2(start, goal, obstacles):
    current, path, hit_point, last_move = start, [start], None, "right"
    while current != goal:
        dx, dy = direction_to_goal(current, goal)
        next_position = (current[0] + dx, current[1] + dy)
        if is_obstacle(next_position, obstacles):
            if not hit_point:
                hit_point = current
            next_position, last_move = follow_boundary(current, obstacles, last_move)
            if can_leave_to_goal(next_position, goal, obstacles):
                hit_point = None
        path.append(next_position)
        current = next_position
    return path

def plot_path(path, obstacles, start, goal):
    plt.figure()
    for (x1, y1), (x2, y2) in obstacles:
        obstacle_x = [x1, x2, x2, x1, x1]
        obstacle_y = [y1, y1, y2, y2, y1]
        plt.plot(obstacle_x, obstacle_y, 'r-', label="Obstacle")
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'b-o', label="Path")
    plt.plot(start[0], start[1], 'go', label="Start")
    plt.plot(goal[0], goal[1], 'ro', label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bug 2 Algorithm")
    plt.legend()
    plt.grid(True)
    plt.show()

start, goal = (0, 0), (20, 20)
obstacles = [((2, 2), (6, 6)), ((7, 7), (14, 14))]
path = bug2(start, goal, obstacles)
plot_path(path, obstacles, start, goal)
