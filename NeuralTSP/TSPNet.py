import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout= 0.2)
        
    def forward(self, x):
        y = self.fc1(x)
        out , _ = self.lstm(y)
        return out


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(Decoder,self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first= True, dropout= 0.2)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout= 0.2, batch_first= True)

    def forward(self, x, enc_outs, h0, c0, indices_to_ignore):
        # LSTM output
        """ x: the vector of the next chosen city, dimension is input_dim
            enc_outs: output of the encoder used in the attention to generate the probability of visiting the next city
            h0,c0: first LSTM latent inputs,
            idices_to_ignor: ignor the visited cities in the attention
        """
        y, (hn, cn) = self.lstm(x, (h0, c0))  # y: (N, 1, d)

        # Create a mask for attention
        # enc_outs: (N, L, d), indices_to_ignore: (N, s)
        N, L, _ = enc_outs.shape
        mask = torch.zeros((N, L), dtype=torch.bool, device=enc_outs.device)  # Initialize mask to False

        # Set True for indices to ignore
        if not indices_to_ignore is None:
            for i in range(N):
                mask[i, indices_to_ignore[i]] = True

        # Apply attention with mask
        _ , attn_weights = self.attn(query=y, key=enc_outs, value=enc_outs, key_padding_mask=mask)  # Masked attention
        attn_weights = attn_weights.squeeze(1)  # (N, L)

        return (hn, cn), attn_weights


class TSPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, Num_L_enc =3, 
                Num_L_dec = 3, num_heads = 2):
        super(TSPNet, self).__init__()
        self.Num_L_dec = Num_L_dec
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim, Num_L_enc).to(device)
        self.decoder = Decoder(hidden_dim, hidden_dim, Num_L_dec, num_heads).to(device)
        self.device = device
    def forward(self, X, mod = 'train'):
        
        """ Given an input city data X (batch_size, num_cities, dimension),
        it returns  outs (batch_size, num_cities+1, num_cities) which is
        pi(a_t|s_t) the policy: probability of visiting the next city
        and action_indices (batch_size, num_cities+1, 1) which is the chosen
        action a_t.
        """

        batch_size, seq_length, _ = X.size()

        encoded_cities = self.encoder(X) # output shape: (batch_size, num_cities, hidden_dim)

        h0,c0 = torch.zeros(self.Num_L_dec, batch_size, self.hidden_dim).to(self.device), torch.zeros(self.Num_L_dec, batch_size, self.hidden_dim).to(self.device)
        
        start_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        x_star = start_token.expand(batch_size, -1, -1).to(self.device)# the first input to the decoder is a vector we have to learn

        outs = torch.zeros(batch_size, seq_length+1, seq_length).to(self.device)
        action_indices = torch.zeros(batch_size, seq_length+1, 1).to(self.device)

        indices_to_ignore = None # for the first input, we can visit all the cities.
        
        for t in range(seq_length+1):
            if t == seq_length:
                indices_to_ignore = indices_to_ignore[:,1:] # now we can return to the first city we visited
            (hn,cn), attn_weights = self.decoder(x_star, encoded_cities, h0,c0, indices_to_ignore)
            attn_weights = torch.clamp(attn_weights, min=1e-9)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True) # (N, L)
            if mod == 'train':
                idx = torch.multinomial(attn_weights, num_samples=1).squeeze(-1) # (N,)
            elif mod == 'eval':
                idx = torch.argmax(attn_weights, dim=-1) # (N,)
            else:
                raise('wrong mode')
            x_star = encoded_cities[torch.arange(batch_size), idx, :].unsqueeze(1) # (N, 1, d)
            outs[:,t,:] = attn_weights
            action_indices[:,t,0] = idx
            h0,c0 = hn,cn
            if t==0:
                indices_to_ignore = idx.unsqueeze(-1) # (N,1)
            else:
                indices_to_ignore = torch.cat((indices_to_ignore, idx.unsqueeze(-1)),dim=-1).long()
            
        return outs, action_indices


if __name__ == "__main__":
    input_dim = 2
    hidden_dim = 128
    num_layers = 2
    num_heads = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TSPNet(input_dim, hidden_dim, device, num_layers, num_layers, num_heads)
    print(model)




