import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess


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
        _, num_heads, _, _ = attn_scores.shape
        for head in range(num_heads):
            if mask is not None:
                attn_scores[:, head, :, :] = attn_scores[:, head, :, :].masked_fill(
                    mask == 0, -1e9
                )

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
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.W_o(attn_output)
        return output


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
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


# Updated EncodersDecoders class (now including decoders)
class EncodersDecoders(nn.Module):
    def __init__(self, n_embed, state_shape, action_shape):
        super(EncodersDecoders, self).__init__()
        # Encoders
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_shape[-3], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * state_shape[-2] * state_shape[-1], n_embed),
            nn.LayerNorm(n_embed),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv2d(action_shape[-3], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * action_shape[-2] * action_shape[-1], n_embed),
            nn.LayerNorm(n_embed),
        )
        self.rtg_encoder = nn.Sequential(
            nn.Linear(1, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.LayerNorm(n_embed),
        )
        self.timesteps_encoder = nn.Sequential(
            nn.Linear(1, n_embed),
            nn.ReLU(),
            nn.Linear(n_embed, n_embed),
            nn.LayerNorm(n_embed),
        )

        # Decoders (no normalization in the decoder output)
        self.action_decoder = nn.Sequential(
            nn.Linear(n_embed, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, action_shape[-3], kernel_size=3, stride=1, padding=1
            ),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(n_embed, 128 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(32, state_shape[-3], kernel_size=3, stride=1, padding=1),
        )


# Modified Transformer class
class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        n_embed,
        state_shape,
        action_shape,
        max_seq_length,
        max_timesteps=200,
        dropout=0.1,
        device="cuda",
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.n_embed = n_embed
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.max_seq_length = max_seq_length

        # Use the updated EncodersDecoders class
        self.encoders_decoders = EncodersDecoders(n_embed, state_shape, action_shape)
        self.encoders_decoders.timesteps_encoder = nn.Embedding(max_timesteps, n_embed)

        # Input projection
        self.input_projection = nn.Linear(n_embed, d_model)

        # Reward projection
        self.reward_projection = nn.Linear(d_model, 1)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length * 3, d_model))

    def forward(self, states, actions, rtgs, timesteps, mask=None):
        batch_size, seq_length = states.shape[:2]

        # Ensure inputs are on the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rtgs = rtgs.to(self.device)
        timesteps = timesteps.to(self.device).long()

        # --- Encode inputs ---
        # Reshape for encoders: (B, L, ...) -> (B*L, ...)
        state_embeddings = self.encoders_decoders.state_encoder(
            states.reshape(-1, *self.state_shape)
        )
        action_embeddings = self.encoders_decoders.action_encoder(
            actions.reshape(-1, *self.action_shape)
        )
        rtg_embeddings = self.encoders_decoders.rtg_encoder(rtgs.reshape(-1, 1))
        # Clamp timesteps before embedding
        timesteps = torch.clamp(
            timesteps, 0, self.encoders_decoders.timesteps_encoder.num_embeddings - 1
        )
        time_embeddings = self.encoders_decoders.timesteps_encoder(
            timesteps.reshape(-1)
        )

        # Reshape back to (B, L, n_embed) or (B, L-1, n_embed)
        state_embeddings = state_embeddings.reshape(
            batch_size, seq_length, self.n_embed
        )
        # Note: actions has length L-1
        action_embeddings = action_embeddings.reshape(
            batch_size, seq_length - 1, self.n_embed
        )
        rtg_embeddings = rtg_embeddings.reshape(batch_size, seq_length, self.n_embed)
        time_embeddings = time_embeddings.reshape(batch_size, seq_length, self.n_embed)

        # --- Create interleaved input sequence for Transformer ---
        # Sequence: (R_0, S_0, A_0, R_1, S_1, A_1, ..., R_{L-1}, S_{L-1})
        # Length: 3 * (L-1) + 2 = 3L - 1

        token_embeddings = torch.zeros(
            (batch_size, seq_length * 3 - 1, self.n_embed),
            dtype=torch.float32,
            device=self.device,
        )

        # Embeddings for R_t, S_t, A_t
        token_embeddings[:, 0::3, :] = rtg_embeddings + time_embeddings
        token_embeddings[:, 1::3, :] = state_embeddings + time_embeddings
        # Actions have length L-1, time embeddings for actions use t=0..L-2
        token_embeddings[:, 2::3, :] = action_embeddings + time_embeddings[:, :-1, :]

        # Project to d_model
        x = self.input_projection(token_embeddings)

        # Add positional embeddings
        # Ensure pos_embed matches the sequence length
        seq_len_actual = x.shape[1]
        x = x + self.pos_embed[:, :seq_len_actual, :]

        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # --- Decode outputs ---
        # We want to predict A_t, S_{t+1}, R_{t+1} based on the sequence up to time t.
        # Predict A_t from output token corresponding to S_t input: x[:, 1::3, :]
        # Predict S_{t+1} from output token corresponding to A_t input: x[:, 2::3, :]
        # Predict R_{t+1} from output token corresponding to A_t input: x[:, 2::3, :]

        # Output tokens corresponding to S_t inputs (predict A_t)
        # Shape: (B, L, d_model)
        action_pred_tokens = x[:, 1::3, :]

        # Output tokens corresponding to A_t inputs (predict S_{t+1} and R_{t+1})
        # Shape: (B, L-1, d_model)
        state_reward_pred_tokens = x[:, 2::3, :]

        # Decode Action Predictions (A_t)
        # Input shape: (B*L, d_model)
        # Output shape: (B, L, C_a, H, W)
        action_preds = self.encoders_decoders.action_decoder(
            action_pred_tokens.reshape(-1, self.d_model)
        ).reshape(batch_size, seq_length, *self.action_shape)

        # Decode State Predictions (S_{t+1})
        # Input shape: (B*(L-1), d_model)
        # Output shape: (B, L-1, C_s, H, W)
        state_preds = self.encoders_decoders.state_decoder(
            state_reward_pred_tokens.reshape(-1, self.d_model)
        ).reshape(batch_size, seq_length - 1, *self.state_shape)

        # Decode Reward Predictions (R_{t+1})
        # Input shape: (B*(L-1), d_model)
        # Output shape: (B, L-1, 1)
        reward_preds = self.reward_projection(state_reward_pred_tokens)

        # Return predictions
        # We need to predict action A_t for t=0..L-1 (L predictions)
        # We need to predict state S_{t+1} for t=0..L-2 (L-1 predictions)
        # We need to predict reward R_{t+1} for t=0..L-2 (L-1 predictions)
        return action_preds, state_preds, reward_preds

    def predict_move(
        self, board, rtg, timestep, encode_board_fn, decode_action_fn, context_window=10
    ):
        self.eval()
        if not self.training:
            self.eval()

        with torch.no_grad():
            # --- Prepare context ---
            board_copy = board.copy()
            move_stack = list(board.move_stack)
            context_moves = (
                move_stack[-(context_window - 1) :] if len(move_stack) > 0 else []
            )
            context_len = len(context_moves)
            seq_len = context_len + 1

            # --- Create tensors ---
            # We need L states, L-1 actions, L RTGs, L timesteps
            states = torch.zeros(
                (1, seq_len, *self.state_shape), device=self.device, dtype=torch.float32
            )
            actions = torch.zeros(
                (1, seq_len - 1, *self.action_shape),
                device=self.device,
                dtype=torch.float32,
            )
            rtgs = torch.full(
                (1, seq_len, 1), rtg, device=self.device, dtype=torch.float32
            )
            timesteps = torch.arange(
                max(0, timestep - context_len), timestep + 1, device=self.device
            )
            timesteps = timesteps.unsqueeze(0)

            # Encode the current board state (S_L)
            states[0, -1] = torch.tensor(encode_board_fn(board), device=self.device)

            # Fill in the context states and actions (backward from S_{L-1}, A_{L-1})
            current_board_for_encoding = board.copy()
            for i in range(context_len):
                move_index = -(i + 1)
                move = context_moves[move_index]

                # Pop the move to get the board state *before* the move
                current_board_for_encoding.pop()

                # Encode the state S_{L-1-i}
                states[0, move_index - 1] = torch.tensor(
                    encode_board_fn(current_board_for_encoding), device=self.device
                )
                # Encode the action A_{L-1-i} (taken from the state encoded above)
                actions[0, move_index] = torch.tensor(
                    self.dataset.encode_move(move, current_board_for_encoding),
                    device=self.device,
                )

            # Generate causal mask for the sequence length (3*L - 1)
            mask = torch.tril(
                torch.ones((seq_len * 3 - 1, seq_len * 3 - 1), device=self.device)
            ).unsqueeze(0)

            # --- Get model prediction ---
            action_preds, _, _ = self.forward(
                states=states,
                actions=actions,
                rtgs=rtgs,
                timesteps=timesteps,
                mask=mask,
            )

            # --- Decode the predicted action ---
            last_action_pred_planes = action_preds[0, -1]

            return decode_action_fn(last_action_pred_planes, board)
