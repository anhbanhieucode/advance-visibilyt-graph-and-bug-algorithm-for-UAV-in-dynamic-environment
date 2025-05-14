import matplotlib.pyplot as plt
import numpy as np
import heapq
import time 
# List of 10 non-overlapping rectangles
'''
rectangles = [
    (54, 63, 13, 7),
    (50, 6, 9, 13),
    (38, 17, 8, 13),
    (59, 13, 6, 14),
    (8, 52, 6, 8),
    (59, 70, 16, 17),
    (35, 49, 12, 8),
    (65, 5, 10, 14),
    (17, 43, 6, 14),
    (61, 39, 9, 17)
]
'''
'''
rectangles = [
    #[2, 2, 4, 3],       # Medium rectangle in bottom-left
    #[10, 2, 6, 2],      # Long horizontal rectangle in lower-middle
    #[20, 2, 2, 4],      # Small vertical rectangle in bottom-right
    #[2, 10, 3, 6],      # Long vertical rectangle in middle-left
    #[10, 10, 4, 4],     # Square in the center
    #[22, 10, 5, 2],     # Thin horizontal rectangle in middle-right
    #[2, 22, 3, 2],      # Small rectangle in top-left
    #[10, 22, 8, 1],     # Long horizontal rectangle in top-middle
    #[22, 22, 4, 3],     # Medium rectangle in top-right
    #[16,8, 2 , 12],
    #[10, 15, 2, 4],
    [34, 1, 5, 12], [46, 1, 9, 7], [58, 1, 8, 8],
    [70, 1, 7, 6], [82, 1, 8, 10], [94, 1, 6, 5],
    [2, 15, 7, 8], [10, 15, 8, 5], [22, 15, 6, 8],
    [34, 15, 12, 3], [50, 15, 5, 7], [58, 15, 7, 6],
    [70, 15, 9, 4], [82, 15, 8, 5], [94, 15, 6, 8],
    [2, 30, 5, 6], [10, 30, 10, 3], [25, 30, 8, 6],
    [34, 30, 4, 10], [46, 30, 5, 7], [58, 30, 6, 5],
    [70, 30, 9, 4], [82, 30, 6, 5], [94, 30, 5, 10],
    [2, 45, 10, 9], [14, 45, 5, 4], [25, 45, 6, 8],
    [34, 45, 8, 5], [46, 45, 7, 6], [58, 45, 5, 7],
    [70, 45, 6, 4], [82, 45, 9, 5], [94, 45, 6, 5],
    [2, 60, 5, 6], [12, 60, 5, 12], [26, 60, 7, 5],
    [34, 60, 4, 9], [46, 60, 11, 3], [58, 60, 6, 8],
    [70, 60, 9, 4], [82, 60, 5, 5], [94, 60, 7, 8],
    [2, 75, 8, 5], [12, 75, 6, 10], [22, 75, 5, 7],
    [34, 75, 7, 4], [46, 75, 10, 5], [58, 75, 5, 8],
    [70, 75, 6, 7], [82, 75, 8, 5], [94, 75, 10, 4],
    [2, 90, 6, 5], [10, 90, 8, 10], [20, 90, 4, 8],
    [30, 90, 12, 3], [45, 90, 5, 6], [58, 90, 7, 4],
    [70, 90, 8, 5], [82, 90, 10, 4], [94, 90, 6, 5]
]
'''

'''
rectangles =  [
      [14,10,50,5],
   [65,10,5,50],
   [14,55,50,5]
]
'''

rectangles =  [
   [4,30,3,30],
   [14,27,3,35],
   [24,28,3,35],
   [34,20,3,38],
   [44,30,3,36],
   [54,26,3,30],
   [64,30,3,30],
   [74,40,3,40],
   [84,30,3,30],
   [94,30,3,35],

   [10,10,10,5],
   [60,5,10,7],
   [20,80,10,8],
   [50,85,15,7],
   [80,14,15,8],
   [4,70,10,5]
   
]

# Define start and goal points
start_point = (0, 50)
goal_point = (50, 50)
bound = 2 #p
robot_radius = 0.9 #R
merge = 1
# Compute rectangle centers
centers = [(x + w / 2, y + h / 2) for x, y, w, h in rectangles]

# Group rectangles based on distance
groups = []
visited = set()

def find_group(idx, group):
    if idx in visited:
        return
    visited.add(idx)
    group.append(idx)
    for j, center in enumerate(centers):
        if j != idx and np.linalg.norm(np.array(centers[idx]) - np.array(center)) < merge:
            find_group(j, group)

for i in range(len(rectangles)):
    if i not in visited:
        group = []
        find_group(i, group)
        groups.append(group)

# Function to check if a point is inside a rectangle
def is_point_in_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh

# Function to check if two line segments (p1, p2) and (p3, p4) intersect
def ccw(A, B, C):
    """Checks if the triplet of points A, B, C makes a counter-clockwise turn."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersects(p1, p2, p3, p4):
    """Returns True if the line segments p1p2 and p3p4 intersect."""
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

# Function to check if a point is inside a rectangle
def is_point_in_rect(px, py, rect):
    """Checks if a point (px, py) is inside the rectangle defined by rect."""
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh

# Function to check if a line segment intersects a rectangle
def line_intersects_rect(x1, y1, x2, y2, rect, R = bound):
    """Returns True if the line segment (x1, y1) -> (x2, y2) intersects the rectangle."""
    rx, ry, rw, rh = rect
    # Define the four corners of the rectangle
    corners = [(rx- R, ry- R), (rx + rw + R, ry - R), (rx - R, ry + rh + R), (rx + rw + R, ry + rh + R)]
    
    # Define the four edges of the rectangle by pairs of corners
    edges = [(corners[i], corners[(i+1)%4]) for i in range(4)]
    
    # Check if the line segment intersects with any of the rectangle's edges
    for (x3, y3), (x4, y4) in edges:
        if line_intersects((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
            return True
    
    # Check if either endpoint is inside the rectangle
    if is_point_in_rect(x1, y1, rect) or is_point_in_rect(x2, y2, rect):
        return True
    
    return False

def line_intersects_rect_for_djistra(x1, y1, x2, y2, rect, R = robot_radius):
    """Returns True if the line segment (x1, y1) -> (x2, y2) intersects the rectangle."""
    rx, ry, rw, rh = rect
    # Define the four corners of the rectangle
    corners = [(rx- R, ry- R), (rx + rw + R, ry - R), (rx - R, ry + rh + R), (rx + rw + R, ry + rh + R)]
    
    # Define the four edges of the rectangle by pairs of corners
    edges = [(corners[i], corners[(i+1)%4]) for i in range(4)]
    
    # Check if the line segment intersects with any of the rectangle's edges
    for (x3, y3), (x4, y4) in edges:
        if line_intersects((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
            return True
    
    # Check if either endpoint is inside the rectangle
    if is_point_in_rect(x1, y1, rect) or is_point_in_rect(x2, y2, rect):
        return True
    
    return False


def add_edges(nodes):
    edges = []
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j:
                x1, y1 = node1
                x2, y2 = node2
                # Check if the line connecting the two nodes intersects any rectangle
                intersects = False
                for rect in rectangles:
                    if line_intersects_rect_for_djistra(x1, y1, x2, y2, rect):
                        intersects = True
                        break
                if not intersects:
                    edges.append((node1, node2))
    return edges
#Djisktra's Algorithm
def dijkstra(start, goal, edges):
    # Step 1: Initialize the priority queue and distance dictionary
    pq = [(0, start)]  # (distance, node)
    distances = {start: 0}
    predecessors = {start: None}
    
    while pq:
        # Step 2: Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(pq)
        
        # If we reached the goal, reconstruct the path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors[current_node]
            return path[::-1]  # Reverse the path to get the correct order
        
        # Step 3: Explore neighbors (connected nodes)
        for neighbor1, neighbor2 in edges:
            if current_node == neighbor1:
                neighbor = neighbor2
            elif current_node == neighbor2:
                neighbor = neighbor1
            else:
                continue
            
            # Calculate the new distance to the neighbor
            new_distance = current_distance + np.linalg.norm(np.array(current_node) - np.array(neighbor))
            
            # If we found a shorter path to the neighbor, update the distance and predecessor
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    
    return None  # If no path is found


# Find rectangles that intersect with the line
intersecting_groups = set()

start_time = time.time()

for group_idx, group in enumerate(groups):
    for rect_idx in group:
        if line_intersects_rect(start_point[0], start_point[1], goal_point[0], goal_point[1], rectangles[rect_idx]):
            intersecting_groups.add(group_idx)
            break

# Plot the rectangles with group-based colors
fig, ax = plt.subplots(figsize=(16,16))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Global Map")
# Assign a unique color to each group
colors = plt.cm.get_cmap("tab10", len(groups))

# List to store the corners (nodes)
nodes = []

# Draw rectangles with group-based colors
for group_idx, group in enumerate(groups):
    color = colors(group_idx)
    for rect_idx in group:
        x, y, width, height = rectangles[rect_idx]
        ax.add_patch(plt.Rectangle((x, y), width, height, color=color, alpha=0.6))

# Plot start and goal points
ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
ax.plot(goal_point[0], goal_point[1], 'ro', markersize=10, label='Goal')

# Add a label for the points
ax.text(start_point[0], start_point[1], ' Start', fontsize=12, ha='right', color='green')
ax.text(goal_point[0], goal_point[1], ' Goal', fontsize=12, ha='left', color='red')

# Plot the corners of the rectangles in the intersecting groups and store them in 'nodes'
for group_idx in intersecting_groups:
    for rect_idx in groups[group_idx]:
        x, y, width, height = rectangles[rect_idx]
        corners = [(x - bound, y - bound), (x + width+ bound, y - bound), (x - bound, y + height + bound), (x + width + bound, y + height + bound)]
        for cx, cy in corners:
            #ax.plot(cx, cy, 'bo', markersize=5)
            nodes.append((cx, cy))  # Add the corner to the nodes list

# Remove points that are inside any rectangle from the 'nodes' list
filtered_nodes = [node for node in nodes if not any(is_point_in_rect(node[0], node[1], rect) for rect in rectangles)]

'''
# Print the stored nodes (corners) after filtering
print("\nFiltered corners (nodes):")
for node in filtered_nodes:
    print(node)
'''
chosen_node = [start_point]
for node in filtered_nodes:
    chosen_node.append(node)
chosen_node.append(goal_point)

# Get the valid edges
edges = add_edges(chosen_node)

# Get the path
path = dijkstra(tuple(start_point), tuple(goal_point), edges)

end_time = time.time()
elapsed_time = end_time-start_time
print("Elapsed time: ", elapsed_time)

# Plot only the remaining points after filtering
for node in filtered_nodes:
    ax.plot(node[0], node[1], 'go', markersize=5)  # Plot only the points that are not inside any rectangle

# Plot the line connecting start and goal points
ax.plot([start_point[0], goal_point[0]], [start_point[1], goal_point[1]], 'k--', linewidth=2, label='Path')

# Plot the edges between the nodes that don't intersect any rectangle
for edge in edges:
    x_values, y_values = zip(*edge)
    ax.plot(x_values, y_values, 'grey', linewidth=1)

# Plot the shortest path
if path:
    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, 'b-', linewidth=2, label="Shortest Path")

# Display the plot
plt.grid(True)
ax.legend()
plt.show()

# Print the intersecting rectangles and their corners
'''
print("Intersecting groups:")
for group_idx in intersecting_groups:
    print(f"Group {group_idx + 1}:")
    for rect_idx in groups[group_idx]:
        x, y, width, height = rectangles[rect_idx]
        corners = [(x, y), (x + width, y), (x, y + height), (x + width, y + height)]
        print(f"  Rectangle {rectangles[rect_idx]} with corners {corners}")
'''

