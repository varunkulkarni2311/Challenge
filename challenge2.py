import numpy as np
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_planets(n, coordinate_range):
    """Generate planets with random coordinates."""
    return np.random.uniform(-coordinate_range, coordinate_range, (n, 3))

def generate_vector(planet1, planet2):
    """Generate vector from planet1 to planet2."""
    return np.array(planet2) - np.array(planet1)
def find_path_towards_target(planets, source_index, target_index):
    source = planets[source_index]
    target = planets[target_index]
    current = source
    path = [current]
    visited_indices = {source_index}

    print(f"Starting at planet {source_index} with coordinates {source}")
    print(f"Target is planet {target_index} with coordinates {target}\n")

    while not np.array_equal(current, target):
        # Find all neighbors
        distances, indices = tree.query(current, k=len(planets))
        # Filter out the current planet and already visited planets
        valid_indices = [i for i in indices if i not in visited_indices and i != source_index]
        
        # Initialize variables to find the minimum valid hop
        min_distance = float('inf')
        next_index = -1
        for i in valid_indices:
            potential_hop = planets[i]
            hop_vector = generate_vector(current, potential_hop)
            target_vector = generate_vector(current, target)
            if np.dot(hop_vector, target_vector) > 0:  # Check for positive component along the target vector
                hop_distance = np.linalg.norm(hop_vector)
                if hop_distance < min_distance:
                    min_distance = hop_distance
                    next_index = i

        if next_index == -1:
            print("No further progress towards the target can be made.")
            break

        # Update current planet, path, and visited indices
        visited_indices.add(next_index)
        current = planets[next_index]
        path.append(current)
        print(f"Hopping from {path[-2]} to {current}, distance: {min_distance:.2f}")

    return np.array(path)
def plot_path_3d(planets, path, source, target):
    """Plot planets and the path in 3D."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot planets
    ax.scatter(planets[:, 0], planets[:, 1], planets[:, 2], color='blue', marker='o', label='Planets')

    # Highlight source and target planets
    ax.scatter(source[0], source[1], source[2], color='yellow', marker='*', s=100, label='Source')
    ax.scatter(target[0], target[1], target[2], color='red', marker='X', s=100, label='Target')
    
    # Plot path
    path_xs, path_ys, path_zs = zip(*path)
    ax.plot(path_xs, path_ys, path_zs, color='red', marker='o', label='Path')
    
    # Plot the main vector from source to target as a straight line
    ax.plot([source[0], target[0]], [source[1], target[1]], [source[2], target[2]], 
            color='green', linestyle='--', linewidth=2, label='Main Vector')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    
    plt.show()

# Parameters
n = 40  # Number of planets
coordinate_range = 100  # Coordinate range for planet positions

# Generate planets
planets = generate_planets(n, coordinate_range)

# Randomly select two different planets as source and target
source_index, target_index = random.sample(range(len(planets)), 2)
source = planets[source_index]
target = planets[target_index]

# Find the path using the k-d tree approach
tree = KDTree(planets)

# Find the safer path using the new approach
safer_path = find_path_towards_target(planets, source_index, target_index)

# Plot with distinct markers for source and target
plot_path_3d(planets, safer_path, source, target)
