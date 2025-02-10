
import torch
from TSPNet import TSPNet
from matplotlib import pyplot as plt
from utils import *
import os
import matplotlib.pyplot as plt
from torchinfo import summary
def plot_routes(cities, routes, index_range, save_folder='plots'):
    """
    Plots the routes for a range of indices and saves the plots in a folder.

    Args:
        cities (torch.Tensor): Tensor of shape (B, N, 2) representing city coordinates.
        routes (torch.Tensor): Tensor of shape (B, N) representing routes.
        index_range (range): Range of indices to plot.
        save_folder (str): Folder to save the plots. Default is 'plots'.
    """
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    for batch_index in index_range:
        # Extract the specific batch
        cities_batch = cities[batch_index].numpy()
        route_batch = routes[batch_index].long().squeeze().numpy()

        # Get coordinates of cities in the order of the route
        ordered_cities = cities_batch[route_batch]

        # Plot cities
        plt.figure(figsize=(8, 6))
        plt.scatter(cities_batch[:, 0], cities_batch[:, 1], color='blue', zorder=2, label='Cities')
        # for i, (x, y) in enumerate(cities_batch):
        #     plt.text(x, y, f'{i}', fontsize=12, ha='right', color='black')

        # Plot the route
        plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], color='red', linestyle='--', zorder=1, label='Route')

        # Highlight start and end points
        plt.scatter(ordered_cities[0, 0], ordered_cities[0, 1], color='green', s=100, label='Start', zorder=3)
        plt.scatter(ordered_cities[-1, 0], ordered_cities[-1, 1], color='purple', s=100, label='End', zorder=3)

        plt.title(f"Route for Batch {batch_index} cost: {route_cost(cities[batch_index:batch_index+1],routes[batch_index:batch_index+1])[0]:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('off')

        # Save the plot to the specified folder
        save_path = os.path.join(save_folder, f"route_batch_{batch_index}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid showing it

    print(f"Plots saved in '{save_folder}'")


if __name__ == "__main__":
    hidden_dim = 128
    num_layers = 2
    num_heads = 2
    input_dim = 2
    num_data = 20
    num_cities = 50
    city_dim = 2
    mod = 'eval_sampling'
    num_samples = 50
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " , device)

    model = TSPNet(input_dim, hidden_dim, device, num_layers, num_layers, num_heads)
    model.load_state_dict(torch.load('Saved models/0.12025_02_02 20_47_49best_model.pth'))
    model.eval()
    print('model loaded')
    print(summary(model))
    data = torch.rand(num_data,num_cities,city_dim).to(device)
    # data = USA_data.to(device)
    num_data , num_cities , city_dim = data.shape
    if mod == 'eval_greedy':
        _, actions = model(data, mod='eval_greedy')
    elif mod == 'eval_sampling':
        all_samples = torch.zeros(num_data, num_cities+1, num_samples).to(device)
        for i in range(num_samples):
            _, actions = model(data, mod='train')
            all_samples[:,:,i:i+1] = actions
        opt_sample_indices = []
        for data_idx in range(num_data):
            costs_for_data = [route_cost(data[data_idx:data_idx+1],all_samples[data_idx:data_idx+1,:,sample_idx]) for sample_idx in range(num_samples)]
            opt_sample_index = torch.argmin(torch.tensor(costs_for_data)).long()
            opt_sample_indices.append(opt_sample_index)
        opt_sample_indices = torch.tensor(opt_sample_indices)
        actions = all_samples[torch.arange(num_data),:,opt_sample_indices]
    plot_routes(data.cpu(),actions.cpu(),torch.arange(len(data)))
