import argparse
import chess
import torch
from torch import nn
from torch.nn import functional as F
from dataloader import ChessDataset, get_chess_dataloader
from model import Transformer
from tqdm import tqdm
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.preprocessing import StandardScaler


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Decision Transformer for Chess")
parser.add_argument("--d_model", type=int, default=256, help="Dimension of the model")
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of attention heads"
)
parser.add_argument(
    "--num_layers", type=int, default=64, help="Number of transformer layers"
)
parser.add_argument(
    "--d_ff", type=int, default=2048, help="Dimension of feed-forward layer"
)
parser.add_argument(
    "--max_seq_length", type=int, default=100, help="Maximum sequence length"
)
parser.add_argument("--vocab_size", type=int, default=1000, help="Size of vocabulary")
parser.add_argument("--n_embed", type=int, default=512, help="Embedding dimension")
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate"
)  # Increased learning rate
args = parser.parse_args()


def normalize_data(data):
    scaler = StandardScaler()
    shape = data.shape
    flattened = data.reshape(-1, shape[-1])
    normalized = scaler.fit_transform(flattened)
    return normalized.reshape(shape)


def generate_mask(timesteps):
    batch_size, seq_length = timesteps.shape[:2]
    mask = torch.tril(torch.ones(seq_length, seq_length))
    mask = mask.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    mask = mask.to(timesteps.device)[:, :-1, :-1]
    return mask


def normalize_batched_tensor(tensor):
    # Compute mean and std across all dimensions except the batch dimension
    mean = tensor.mean(dim=list(range(1, tensor.dim())), keepdim=True)
    std = tensor.std(dim=list(range(1, tensor.dim())), keepdim=True)
    # Handle cases where std is 0 to avoid division by zero
    std = torch.clamp(std, min=1e-8)
    return (tensor - mean) / std


def train_epoch(model, dataloader, optimizer, device, window_size=10):
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    num_batches = 0

    for data in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        R, s, a, t = [
            d.to(device)
            for d in [
                data["returns_to_go"],
                data["states"],
                data["actions"][:, :window_size],
                data["timesteps"],
            ]
        ]
        R_next, s_next, a_next, t_next = [
            d.to(device)
            for d in [
                data["returns_to_go"][:, 1:],
                data["states"],
                data["actions"],
                data["timesteps"][:, 1:],
            ]
        ]

        a_pred, s_pred, r_pred = model(
            states=s, actions=a, rtgs=R, timesteps=t, mask=generate_mask(t)
        )

        r_pred_norm = normalize_batched_tensor(r_pred)
        R_next_norm = normalize_batched_tensor(R_next)
        s_pred_norm = normalize_batched_tensor(s_pred)
        s_next_norm = normalize_batched_tensor(s_next)

        action_loss = F.cross_entropy(
            a_pred.view(-1, 73, 8, 8), a_next.view(-1, 73, 8, 8)
        )
        state_loss = F.mse_loss(s_pred_norm, s_next_norm)
        reward_loss = F.mse_loss(r_pred_norm, R_next_norm)
        stability_weight = 0.0001
        loss = action_loss * 1000 + stability_weight * (state_loss + reward_loss)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Adjusted clip value

        optimizer.step()

        total_loss += loss.item() / 1000  # Unscale the loss for logging
        total_action_loss += action_loss.item()
        total_state_loss += state_loss.item()
        total_reward_loss += reward_loss.item()
        num_batches += 1

        # Gradient logging (every 100 batches)
        if num_batches % 100 == 0:
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

    return {
        "loss": total_loss / num_batches,
        "action_loss": total_action_loss / num_batches,
        "state_loss": total_state_loss / num_batches,
        "reward_loss": total_reward_loss / num_batches,
        "final_gradient_norm": total_norm,
    }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_action_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", leave=False):
            R, s, a, t = [
                d.to(device)
                for d in [
                    data["returns_to_go"],
                    data["states"],
                    data["actions"][:, :10],
                    data["timesteps"],
                ]
            ]
            R_next, s_next, a_next, t_next = [
                d.to(device)
                for d in [
                    data["returns_to_go"][:, 1:],
                    data["states"],
                    data["actions"],
                    data["timesteps"][:, 1:],
                ]
            ]

            a_pred, s_pred, r_pred = model(
                states=s, actions=a, rtgs=R, timesteps=t, mask=generate_mask(t)
            )

            r_pred_norm = normalize_batched_tensor(r_pred)
            R_next_norm = normalize_batched_tensor(R_next)
            s_pred_norm = normalize_batched_tensor(s_pred)
            s_next_norm = normalize_batched_tensor(s_next)

            action_loss = F.cross_entropy(
                a_pred.view(-1, 73, 8, 8), a_next.view(-1, 73, 8, 8)
            )
            state_loss = F.mse_loss(s_pred_norm, s_next_norm)
            reward_loss = F.mse_loss(r_pred_norm, R_next_norm)
            stability_weight = 0.0001
            loss = action_loss * 1000 + stability_weight * (state_loss + reward_loss)

            total_loss += loss.item() / 1000  # Unscale the loss for logging
            total_action_loss += action_loss.item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "action_loss": total_action_loss / num_batches,
        "state_loss": total_state_loss / num_batches,
        "reward_loss": total_reward_loss / num_batches,
    }


def train_decision_transformer(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
):
    best_val_loss = float("inf")

    # Open CSV file for writing
    with open("loss_log.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["epoch", "type", "loss", "action_loss", "lr"]
        )  # Added action_loss to header

        for epoch in range(num_epochs):
            train_metrics = train_epoch(model, train_dataloader, optimizer, device)
            val_metrics = evaluate(model, val_dataloader, device)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Action Loss: {train_metrics['action_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Action Loss: {val_metrics['action_loss']:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Write train and validation losses to CSV
            csvwriter.writerow(
                [
                    epoch + 1,
                    "train",
                    f"{train_metrics['loss']:.4f}",
                    f"{train_metrics['action_loss']:.4f}",
                    f"{optimizer.param_groups[0]['lr']:.6f}",
                ]
            )
            csvwriter.writerow(
                [
                    epoch + 1,
                    "validation",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['action_loss']:.4f}",
                    f"{optimizer.param_groups[0]['lr']:.6f}",
                ]
            )
            csvfile.flush()

            scheduler.step(val_metrics["loss"])  # Update learning rate

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), "best_model.pth")
                print("  New best model saved!")

            if (epoch + 1) % 10 == 0:  # Test every 10 epochs
                model.load_state_dict(torch.load("best_model.pth"))
                test_metrics = evaluate(model, test_dataloader, device)
                print(f"Test Loss: {test_metrics['loss']:.4f}")
                print(f"Test Action Loss: {test_metrics['action_loss']:.4f}")
                csvwriter.writerow(
                    [
                        epoch + 1,
                        "test",
                        f"{test_metrics['loss']:.4f}",
                        f"{test_metrics['action_loss']:.4f}",
                        f"{optimizer.param_groups[0]['lr']:.6f}",
                    ]
                )
                csvfile.flush()


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Transformer(
    d_model=512,  # Increased from 256
    num_heads=8,
    num_layers=4,  # Reduced from 6
    d_ff=2048,  # Increased from 1024
    n_embed=512,  # Increased from 256
    state_shape=(87, 8, 8),
    action_shape=(73, 8, 8),
    dropout=0.1,
    device=device,
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Increased learning rate
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

# Get dataloaders
train_dataloader, val_dataloader, test_dataloader = get_chess_dataloader(
    num_games=1000,
    max_moves=100,
    trajectory_length=10,
    batch_size=256,
    engine_path="/home/linuxbrew/.linuxbrew/bin/stockfish",
    save_path="chess_dataset.pkl",
    load_path="chess_dataset.pkl",
)

print("Example game has been generated and saved to 'example_game.pgn'")

# Train the model
train_decision_transformer(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs=100,
    device=device,
)
