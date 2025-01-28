
from utils import route_cost, generate_unit_circle_cities
from TSPNet import TSPNet
from data_loader_script import create_data_loader
from train import train
import torch
from datetime import datetime
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import argparse

mp.set_start_method('spawn', force=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="Running Neural TSP Learning")
parser.add_argument("--alpha", type=float, default=0.9, help="EMA rate (default: 0.9)")
parser.add_argument('--use_base', 
                        type=str2bool, 
                        default=True, 
                        help="Enable using base (true/false). Default is True.")
args = parser.parse_args()

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
    num_episodes = 100000
    num_samples = batch_size * num_episodes
    num_cities = 50
    input_dim = 2
    num_workers = 8  #

    data_loader = create_data_loader(batch_size, num_samples, num_cities, input_dim, num_workers=num_workers)
    preloaded_batches = preload_data(data_loader, device)

    run_name = 'TSP/' + str(batch_size) + '_' + str(num_cities) + '_' + str(num_samples) + '_' + '/ANN/'+datetime.now().strftime(("%Y_%m_%d %H_%M_%S"))+ str(args.use_base) + str(args.alpha)
    writer = SummaryWriter(log_dir=run_name)


    hidden_dim = 128
    num_layers = 2
    num_heads = 1

    model = TSPNet(input_dim, hidden_dim, device, num_layers, num_layers, num_heads)
    print(model.device)
    

    trained = train(model, preloaded_batches, writer, lr, use_base = args.use_base, alpha=args.alpha)


