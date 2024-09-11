import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from model import Transformer
import dataloader
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train Decision Transformer for Chess')
parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of feed-forward layer')
parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum sequence length')
parser.add_argument('--vocab_size', type=int, default=1000, help='Size of vocabulary')
parser.add_argument('--n_embed', type=int, default=256, help='Embedding dimension')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
args = parser.parse_args()

# Training loop
def train_decision_transformer(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for R, s, a, t in dataloader:  # dims: (batch_size, K, dim)
            optimizer.zero_grad()
            
            a_preds = model(R, s, a, t)
            loss = F.mse_loss(a_preds, a)  # L2 loss for continuous actions
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Initialize model with command-line arguments
model = Transformer(
    d_model=args.d_model,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    d_ff=args.d_ff,
    max_seq_length=args.max_seq_length,
    vocab_size=args.vocab_size,
    n_embed=args.n_embed
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
dataloader = dataloader.get_chess_dataloader(trajectory_length=10)

# Train the model
train_decision_transformer(model, dataloader, optimizer, num_epochs=args.num_epochs)

