import torch
import torch.nn as nn
import math
# torch.set_default_dtype(torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=500):
        """
        By using (1, max_len, d_model), the positional encoding can be added directly to the input embeddings
        of shape (batch_size, seq_len, d_model) without requiring additional reshaping or computation.
        PyTorch automatically broadcasts the 1 in the batch dimension to match the batch size of the input.
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding with explicit dtype=torch.float32
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        # Compute sine and cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and move to the specified device
        pe = pe.unsqueeze(0).to(device)  # Shape: (1, max_len, d_model)
        # print('peeeeeee',pe.dtype)
        # Register as a buffer (non-trainable but saved with the model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :] * 0.1 # Slice to match sequence length
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, device, dropout=0.3, use_PE=True):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, device)
        self.use_PE = use_PE  # Flag to control positional encoding usage
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4, # feedforward consists of two layers, and the last layer brings the dim back to hidden_dim
            dropout=dropout,
            batch_first=True,
            # norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers).to(device)
        
    def forward(self, x):
        x = self.fc1(x)
        if self.use_PE:
            x = self.positional_encoding(x)  # Add positional encoding if use_PE is True
        out = self.transformer_encoder(x).float()
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, device, dropout=0.3):
        super(Decoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            # norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers).to(device)
        self.transformer_decoder = self.transformer_decoder.float() 
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """ 1. tgt (Target Sequence)
            Description : The input sequence to the decoder (queries in self-attention).
            Parallel to Tutorial : In the tutorial, this corresponds to the "target sequence" Y during training or the previously generated tokens Yt−1
            during inference.
            Shape : (batch_size, target_sequence_length, hidden_dim).
            2. memory (Encoder Output)
            Description : The encoded representation of the input sequence from the encoder (keys and values in cross-attention).
            Parallel to Tutorial : This corresponds to X_hat the output of the encoder.
            Shape : (batch_size, input_sequence_length, hidden_dim). 
            3. tgt_mask (Target Mask)
            Description : A mask applied to the target sequence to enforce causality (prevent attending to future tokens).
            Parallel to Tutorial : This corresponds to the "causal mask" mentioned in the tutorial. It ensures that each position in the target sequence can only attend to itself and prior positions.
            Shape : (target_sequence_length, target_sequence_length).
            4. memory_mask (Memory Mask)
            Description : A mask applied to the encoder output (memory). This is rarely used in practice but can be helpful in certain scenarios (e.g., masking out irrelevant parts of the input).
            Parallel to Tutorial : Not explicitly discussed in the tutorial, but it would correspond to any additional masking applied to the encoder output.
            5. tgt_key_padding_mask (Target Padding Mask)
            Description : A mask to ignore padding tokens in the target sequence.
            Parallel to Tutorial : This corresponds to ignoring padded tokens in the target sequence (if applicable). For example, if the target sequence is shorter than the maximum length, padding tokens are ignored.
            Shape : (batch_size, target_sequence_length). 
            6. memory_key_padding_mask (Memory Padding Mask)
            Description : A mask to ignore padding tokens in the encoder output (memory).
            Parallel to Tutorial : This corresponds to ignoring padded tokens in the input sequence (if applicable). For example, if the input sequence is shorter than the maximum length, padding tokens are ignored.
            Shape : (batch_size, input_sequence_length)."""
    
        out = self.transformer_decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True
        ) # returns the transformed values of tgt after cross-attention
        return out

class TSPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_enc_layers=3, num_dec_layers=3, num_heads=8, dropout=0.3, use_PE=True):
        super(TSPNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        # Encoder
        self.encoder = Encoder(input_dim, hidden_dim, num_enc_layers, num_heads, dropout, use_PE).to(device)

        # Decoder
        self.decoder = Decoder(hidden_dim, num_dec_layers, num_heads, dropout).to(device)

        # Start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)).to(device)

        # Output projection

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, X, mod='train'):
        """
        Given an input city data X (batch_size, num_cities, dimension),
        it returns outs (batch_size, num_cities+1, num_cities) which is
        pi(a_t|s_t) the policy: probability of visiting the next city,
        and action_indices (batch_size, num_cities+1, 1) which is the chosen
        action a_t.
        """
        batch_size, seq_length, _ = X.size()
        
        # Encode the input cities
        encoded_cities = self.encoder(X)  # (batch_size, num_cities, hidden_dim)
        # Initialize variables
        start_token = self.start_token.expand(batch_size, -1, -1).to(self.device)  # (batch_size, 1, hidden_dim)
        outs = torch.zeros(batch_size, seq_length + 1, seq_length).to(self.device)
        action_indices = torch.zeros(batch_size, seq_length + 1, 1).to(self.device)
        indices_to_ignore = None

        # Prepare masks
        # tgt_mask = self.generate_square_subsequent_mask(seq_length + 1).to(self.device)
        
        for t in range(seq_length + 1):  # Loop over all steps, including the last one
            if t == 0:
                tgt = start_token
            else:
                prev_chosen_cities = action_indices[:, :t, 0].long()
                # Select previously chosen cities
                selected_cities = encoded_cities[torch.arange(batch_size).unsqueeze(1), prev_chosen_cities, :]
                tgt = selected_cities  # (batch_size, t, hidden_dim)

            # Apply positional encoding to the target sequence
            tgt = PositionalEncoding(self.hidden_dim, self.device)(tgt)

            # Create memory key padding mask to ignore visited cities
            memory_key_padding_mask = None
            if indices_to_ignore is not None:
                memory_key_padding_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=self.device)
                for i in range(batch_size):
                    memory_key_padding_mask[i, indices_to_ignore[i]] = True

            if t == seq_length:
                # Remove the first visited city from indices_to_ignore
                # print(f"Memory Key Padding Mask Before: {memory_key_padding_mask}")
                memory_key_padding_mask[torch.arange(batch_size), indices_to_ignore[:, 0]] = False
                # print(f"Memory Key Padding Mask After: {memory_key_padding_mask}")

            # Generate causal mask for the target sequence
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

            # Decode
            dec_out = self.decoder(
                tgt=tgt,
                memory=encoded_cities,
                tgt_mask=tgt_mask[:t+1, :t+1],  # Causal mask for the current step
                memory_key_padding_mask=memory_key_padding_mask
            )

            # Use scaled dot-product attention to compute attention scores over encoded cities
            query = dec_out[:, -1, :]  # Query: Last output of the decoder (batch_size, hidden_dim)
            keys = encoded_cities  # Keys: Encoded cities (batch_size, num_cities, hidden_dim)

            # Compute attention scores
            scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)  # (batch_size, 1, num_cities)
            scores = scores.squeeze(1)  # (batch_size, num_cities)
            

            # Apply memory key padding mask to scores
            if memory_key_padding_mask is not None:
                scores = scores.masked_fill(memory_key_padding_mask, float('-inf'))

            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_cities)
            attn_weights = torch.clamp(attn_weights, min=1e-9)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

            # Sample or choose the next city
            if mod == 'train':
                idx = torch.multinomial(attn_weights, num_samples=1).squeeze(-1)  # (batch_size,)
            elif mod == 'eval_greedy':
                idx = torch.argmax(attn_weights, dim=-1)  # (batch_size,)
            else:
                raise ValueError("Invalid mode")

            # Update outputs
            outs[:, t, :] = attn_weights
            action_indices[:, t, 0] = idx

            # Update indices_to_ignore
            if t == 0:
                indices_to_ignore = idx.unsqueeze(-1)  # (batch_size, 1)
            else:
                indices_to_ignore = torch.cat((indices_to_ignore, idx.unsqueeze(-1)), dim=-1).long()
            # print(f"idx: {idx}")
            # print(f"indices_to_ignore: {indices_to_ignore}")
        # print(action_indices[0,:,0])
        outs = torch.clamp(outs, min=1e-9, max=1.0)
        return outs, action_indices