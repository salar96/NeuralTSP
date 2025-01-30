from tqdm import tqdm
import torch
from torch import optim
from utils import route_cost
from datetime import datetime
from torch_optimizer import RAdam



def train(model, preloaded_batches, writer, lr = 0.001, len_print = 100, use_base = True, alpha = 0.9):
    device = model.device
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = RAdam(model.parameters(), lr=lr)
    print(" Training Started with Base and alpha: ",use_base , alpha)
    batch_means = torch.zeros(len_print)
    best_mean_cost = float('inf')  # Initialize with infinity
    best_model_state = None
    Base = torch.tensor([0.0],device = device)
    
    for episode, data_batch in tqdm(enumerate(preloaded_batches)):
        #data_batch = data_batch.to(device, non_blocking=True)
        outs, actions = model(data_batch, mod = 'train')
        
        sum_log_prob = torch.log(outs.gather(2, actions.long()).squeeze(-1)).sum(dim=1)
        costs = route_cost(data_batch, actions)

        batch_means[episode % len_print] = costs.mean().item()
        
        if use_base is True:
          policy_loss = torch.sum(sum_log_prob * (costs - Base)) / len(preloaded_batches)
        else:
          policy_loss = torch.sum(sum_log_prob * costs) / len(preloaded_batches)
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if episode % len_print == len_print - 1:
            mean_cost = batch_means.mean().item()
            if alpha == -1:
              alpha_t = episode/(episode+1)
              Base = alpha_t * Base + (1-alpha_t) * mean_cost
              
            else:
              Base = alpha * Base + (1-alpha) * mean_cost
            print(f"Episode: {episode+1} Mean cost: {mean_cost:.2f}")
            writer.add_scalar('Mean cost', mean_cost, episode)
            if mean_cost < best_mean_cost:
                best_mean_cost = mean_cost
                best_model_state = model.state_dict()
    if best_model_state is not None:
        torch.save(best_model_state, 'Saved models/' + str(alpha) + datetime.now().strftime(("%Y_%m_%d %H_%M_%S")) + 'best_model.pth')
        print(f"Best model saved with mean cost: {best_mean_cost:.2f}")
    writer.close()
    torch.save(model.state_dict(), 'Saved models/' + str(alpha) + datetime.now().strftime(("%Y_%m_%d %H_%M_%S")) + 'last_model.pth')
    print("Training Finished")
    return model

if __name__ == "__main__":
    print("test")