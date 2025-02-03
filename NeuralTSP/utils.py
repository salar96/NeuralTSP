import math
import torch


def route_cost(cities, routes):
    B, N, _ = cities.shape
    routes = routes.squeeze(-1).long()  # Convert to long for indexing
    ordered_cities = cities[torch.arange(B).unsqueeze(1), routes]  # Reorder cities based on routes
    diffs = ordered_cities[:, :-1] - ordered_cities[:, 1:]  # Compute differences between consecutive cities
    distances = torch.norm(diffs, p=2, dim=2)  # Euclidean distances
    total_distances = distances.sum(dim=1)  # Sum distances for each batch
    return total_distances

def generate_unit_circle_cities(B, N, d):
    """
    Generates a PyTorch tensor of size (B, N, d), representing B batches
    of N cities in d-dimensional space, where cities are randomly placed on the unit circle.
    
    Args:
        B (int): Number of batches.
        N (int): Number of cities in each batch.
        d (int): Number of dimensions (must be at least 2, higher dimensions will have zeros).
        
    Returns:
        torch.Tensor: A tensor of shape (B, N, d) with cities on the unit circle.
    """
    if d < 2:
        raise ValueError("Dimension 'd' must be at least 2.")

    # Generate random angles for each city
    angles = torch.rand(B, N) * 2 * math.pi  # Random angles in radians

    # Coordinates on the unit circle
    x_coords = torch.cos(angles)
    y_coords = torch.sin(angles)

    # Create a tensor of zeros for higher dimensions if d > 2
    higher_dims = torch.zeros(B, N, d - 2)

    # Combine x, y, and higher dimensions
    unit_circle_coords = torch.stack((x_coords, y_coords), dim=-1)
    result = torch.cat((unit_circle_coords, higher_dims), dim=-1)
    result[:,0,:] = result[:,-1,:]
    return result






coordinates = [
    [6734, 1453],
    [2233, 10],
    [5530, 1424],
    [401, 841],
    [3082, 1644],
    [7608, 4458],
    [7573, 3716],
    [7265, 1268],
    [6898, 1885],
    [1112, 2049],
    [5468, 2606],
    [5989, 2873],
    [4706, 2674],
    [4612, 2035],
    [6347, 2683],
    [6107, 669],
    [7611, 5184],
    [7462, 3590],
    [7732, 4723],
    [5900, 3561],
    [4483, 3369],
    [6101, 1110],
    [5199, 2182],
    [1633, 2809],
    [4307, 2322],
    [675, 1006],
    [7555, 4819],
    [7541, 3981],
    [3177, 756],
    [7352, 4506],
    [7545, 2801],
    [3245, 3305],
    [6426, 3173],
    [4608, 1198],
    [23, 2216],
    [7248, 3779],
    [7762, 4595],
    [7392, 2244],
    [3484, 2829],
    [6271, 2135],
    [4985, 140],
    [1916, 1569],
    [7280, 4899],
    [7509, 3239],
    [10, 2676],
    [6807, 2993],
    [5185, 3258],
    [3023, 1942]
]

# Convert the list to a PyTorch tensor
USA_data = torch.tensor(coordinates, dtype=torch.float).unsqueeze(0)
# 7762 is the max
USA_data = USA_data / torch.max(USA_data)


    