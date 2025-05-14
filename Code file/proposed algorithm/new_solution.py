import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString, box

def plot_map(map_size, rectangles, start, goal):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_xticks(np.arange(0, map_size+1, 10))
    ax.set_yticks(np.arange(0, map_size+1, 10))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Create line segment from start to goal
    path_line = LineString([start, goal])
    graph = nx.Graph()
    marked_vertices = []
    marked_rects = []
    
    for rect in rectangles:
        x, y, width, height = rect
        rect_box = box(x, y, x + width, y + height)
        intersects = path_line.intersects(rect_box)
        color = 'black' if not intersects else 'red'
        ax.add_patch(plt.Rectangle((x, y), width, height, color=color))
        
        # If the rectangle is marked (intersects path), plot its vertices
        if intersects:
            vertices = [(x-p, y-p), (x + width + p , y - p), (x + width + p, y + height + p), (x - p, y + height + p)]
            for vx, vy in vertices:
                ax.plot(vx, vy, 'bo', markersize=5)  # Mark vertices in blue
                marked_vertices.append((vx, vy))
            marked_rects.append(rect_box)


    # Create a graph from the marked vertices
    for i, v1 in enumerate(marked_vertices):
        for j, v2 in enumerate(marked_vertices):
            if i != j:
                edge_line = LineString([v1, v2])
                # Check if edge intersects any marked rectangle
                if not any(edge_line.intersects(rect) for rect in marked_rects):
                    graph.add_edge(v1, v2, weight=np.linalg.norm(np.array(v1) - np.array(v2)))
                                   
      
    # Draw the graph edges
    for edge in graph.edges:
        v1, v2 = edge
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'c-', linewidth=1)
    
    # Plot start and goal points
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Draw path line
    ax.plot([start[0], goal[0]], [start[1], goal[1]], 'b--', linewidth=2, label='Path')
    
    ax.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Define map size, rectangles, start and goal points
map_size = 100
p = 0.5
rectangles = [
    (10, 10, 15, 10),
    (30, 40, 20, 15),
    (60, 20, 10, 25),
    (50, 70, 18, 12),
    (80, 80, 12, 18),
    (20, 60, 22, 14),
    (40, 10, 16, 20),
    (70, 50, 14, 16)
]
start = (5, 5)
goal = (95, 95)

plot_map(map_size, rectangles, start, goal)