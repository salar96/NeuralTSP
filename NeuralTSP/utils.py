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





if __name__ == "__main__":
    cities = generate_unit_circle_cities(10,10,2)
    routes = torch.randint(10, (10,11))
    