import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        # Linear projections for Query, Key, Value, and Output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    # Scaled Dot-Product Attention
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

# Full Transformer model
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, n_embed, state_shape=(14, 8, 8), action_shape=(73, 8, 8), dropout=0.1, device='cuda'):
        super(Transformer, self).__init__()
        self.device = device
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_embed = n_embed
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Encoders
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_shape[-3], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * state_shape[-2] * state_shape[-1], n_embed)
        )
        self.action_encoder = nn.Sequential(
            nn.Conv2d(action_shape[-3], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * action_shape[-2] * action_shape[-1], n_embed)
        )
        
        # RTG and timestep encoders
        self.rtg_encoder = nn.Sequential(
            nn.Linear(1, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed)
        )
        self.timesteps_encoder = nn.Sequential(
            nn.Linear(1, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed)
        )

        # Input projection
        self.input_projection = nn.Linear(n_embed, d_model)

        # Decoders
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, action_shape[-3], kernel_size=3, stride=1, padding=1),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(d_model, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, state_shape[-3], kernel_size=3, stride=1, padding=1),
        )
        self.reward_projection = nn.Linear(d_model, 1)
    def forward(self, states, rtgs, timesteps, actions=None, mask=None):
        batch_size, seq_length = states.shape[:2]
        states_embeddings = self.state_encoder(states.reshape(batch_size * seq_length, *states.shape[2:])).reshape(batch_size, seq_length, -1)
        rtg_embeddings = self.rtg_encoder(rtgs.unsqueeze(-1).reshape(batch_size * seq_length, *rtgs.shape[2:])).reshape(batch_size, seq_length, -1)
        timesteps_embeddings = self.timesteps_encoder(timesteps.float().unsqueeze(-1).reshape(batch_size * seq_length, *timesteps.shape[2:])).reshape(batch_size, seq_length, -1)

        if actions is not None:
            actions_embeddings = self.action_encoder(actions.reshape(-1, *actions.shape[2:])).reshape(batch_size, seq_length, -1)
            # Combine embeddings with actions
            combined_embeddings = torch.zeros(batch_size, seq_length * 3, self.n_embed, device=self.device)
            combined_embeddings[:, 0::3, :] = rtg_embeddings + timesteps_embeddings
            combined_embeddings[:, 1::3, :] = states_embeddings + timesteps_embeddings
            combined_embeddings[:, 2::3, :] = actions_embeddings + timesteps_embeddings
        else:
            # Combine embeddings without actions
            combined_embeddings = torch.zeros(batch_size, seq_length * 2, self.n_embed, device=self.device)
            combined_embeddings[:, 0::2, :] = rtg_embeddings + timesteps_embeddings
            combined_embeddings[:, 1::2, :] = states_embeddings + timesteps_embeddings

        x = self.input_projection(combined_embeddings)
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Reshape x to separate state, action, and reward
        x = x.view(batch_size, seq_length, 3, self.d_model)
        x = x.transpose(1, 2)  # Now shape is (batch_size, 3, seq_length, d_model)
        
        if actions is not None:
            # Separate x into state, action, and reward
            x_state = x[:, 0]  # (batch_size, seq_length, d_model)
            x_action = x[:, 1]  # (batch_size, seq_length, d_model)
            x_reward = x[:, 2]  # (batch_size, seq_length, d_model)
 
            # Apply decoders
            action_preds = self.action_decoder(x_action.view(-1, self.d_model)).view(batch_size, seq_length, *self.action_shape)
            state_preds = self.state_decoder(x_state.view(-1, self.d_model)).view(batch_size, seq_length, *self.state_shape)
            reward_preds = self.reward_projection(x_reward)

            return action_preds, state_preds, reward_preds
        else:
            x_reward = x[:, 0]  # (batch_size, seq_length, d_model)
            x_state = x[:, 1]  # (batch_size, seq_length, d_model)

            # Apply decoders
            state_preds = self.state_decoder(x_state.view(-1, self.d_model)).view(batch_size, seq_length, *self.state_shape)
            reward_preds = self.reward_projection(x_reward)

            return None, state_preds, reward_preds

