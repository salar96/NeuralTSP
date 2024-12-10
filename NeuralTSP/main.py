
from utils import route_cost, generate_unit_circle_cities
from TSPNet import TSPNet
from data_loader_script import create_data_loader
from train import train
import torch
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

mp.set_start_method('spawn', force=True)

def preload_data(data_loader, device):
    preloaded_batches = []
    for data_batch in data_loader:
        preloaded_batches.append(data_batch.to(device, non_blocking=True))
    return preloaded_batches

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " , device)

    lr = 0.001
    batch_size = 128
    num_episodes = 50000
    num_samples = batch_size * num_episodes
    num_cities = 10
    input_dim = 2
    num_workers = 8  #

    data_loader = create_data_loader(batch_size, num_samples, num_cities, input_dim, num_workers=num_workers)
    preloaded_batches = preload_data(data_loader, device)

    run_name = 'TSP/' + str(batch_size) + '_' + str(num_cities) + '_' + str(num_samples) + '_' + '/ANN/'+datetime.now().strftime(("%Y_%m_%d %H_%M_%S"))
    writer = SummaryWriter(log_dir=run_name)


    hidden_dim = 128
    num_layers = 2
    num_heads = 1

    model = TSPNet(input_dim, hidden_dim, device, num_layers, num_layers, num_heads)
    print(model.device)
    print(summary(model))

    trained = train(model, preloaded_batches, writer, lr)


