from tqdm import tqdm
import torch
from torch import optim
from utils import *
from datetime import datetime
from torch_optimizer import RAdam
import copy



def train(model, data_loader, eval_loader, writer, lr = 0.001, len_print = 100, check_params = False):
    device = model.device
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = RAdam(model.parameters(), lr=lr)
    print(" Training Started with NN Baseline.")
    
  
    Base_model = model.__class__(**model.init_args)
    Base_model.load_state_dict(model.state_dict())
    Base_model.eval()
    len_eval = len(eval_loader)
    batch_means = torch.zeros(len_eval)
    batch_means_B = torch.zeros(len_eval)
    Overfit_trig = 0
    for episode, data_batch in tqdm(enumerate(data_loader)):
        data_batch = data_batch.to(device, non_blocking=True)
        batch_size = data_batch.shape[0]

        outs, actions = model(data_batch, mod = 'train')
        _, actions_B = Base_model(data_batch, mod = 'eval_greedy')
        
        costs = route_cost(data_batch, actions) #f
        costs_B = route_cost(data_batch, actions_B)
        
        sum_log_prob = torch.log(outs.gather(2, actions.long()).squeeze(-1)).sum(dim=1) #log(p_theta)

        policy_loss = torch.sum(sum_log_prob * (costs - costs_B)) / batch_size
   
        
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if check_params:
              check_model_weights(model)
              check_gradients(model)
        
        if episode % len_print == len_print - 1:
            
            for eval_episode, eval_batch in enumerate(eval_loader): 
                eval_batch = eval_batch.to(device, non_blocking=True)
                _, actions_eval = model(eval_batch, mod = 'eval_greedy')
                _, actions_B_eval = Base_model(eval_batch, mod = 'eval_greedy')
                
                costs_eval = route_cost(eval_batch, actions_eval) 
                costs_B_eval = route_cost(eval_batch, actions_B_eval)
                batch_means[eval_episode] = costs_eval.mean().item()
                batch_means_B[eval_episode] = costs_B_eval.mean().item()
                  
            mean_cost = batch_means.mean().item()
            print(f"Episode: {episode+1} Mean cost: {mean_cost:.2f}")
            writer.add_scalar('Mean cost', mean_cost, episode)
            if is_paired_ttest_significant(batch_means,batch_means_B ,0.1):
                Base_model.load_state_dict(model.state_dict())
                Base_model.eval()
                print("Base model updated")


    torch.save(Base_model.state_dict(), 'Saved models/' + "Baseline" + datetime.now().strftime(("%Y_%m_%d %H_%M_%S")) + 'best_model.pth')
    print(f"Best model saved with mean cost: {batch_means_B.mean().item():.2f}")
    writer.close()
    return model

if __name__ == "__main__":
    print("test")