import torch
from torch import nn
from torch.nn import functional as F
from dataloader import get_chess_dataloader
from model import Transformer
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0  # Initialize total_loss here
    total_action_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    num_batches = 0

    for data in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        R, s, a, t = [
            d.to(device)
            for d in [
                data["returns_to_go"][:, :10],
                data["states"][:, :10],
                data["actions"][:, :10],
                data["timesteps"][:, :10],
            ]
        ]
        R_next, s_next, a_next, t_next = [
            d.to(device)
            for d in [
                data["returns_to_go"][:, 1:],
                data["states"][:, 1:],
                data["actions"][:, 1:],
                data["timesteps"][:, 1:],
            ]
        ]

        a_pred, s_pred, r_pred = model(states=s, actions=a, rtgs=R, timesteps=t)
        action_loss = F.mse_loss(a_pred, a_next)
        state_loss = F.mse_loss(s_pred, s_next)
        reward_loss = F.mse_loss(r_pred, R_next)

        stability_weight = 0.001
        loss = action_loss + stability_weight * (state_loss + reward_loss)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
                    data["returns_to_go"][:, :10],
                    data["states"][:, :10],
                    data["actions"][:, :10],
                    data["timesteps"][:, :10],
                ]
            ]
            R_next, s_next, a_next, t_next = [
                d.to(device)
                for d in [
                    data["returns_to_go"][:, 1:],
                    data["states"][:, 1:],
                    data["actions"][:, 1:],
                    data["timesteps"][:, 1:],
                ]
            ]

            a_pred, s_pred, r_pred = model(states=s, actions=a, rtgs=R, timesteps=t)
            action_loss = F.mse_loss(a_pred, a_next)
            state_loss = F.mse_loss(s_pred, s_next)
            reward_loss = F.mse_loss(r_pred, R_next)

            stability_weight = 0.001
            loss = action_loss + stability_weight * (state_loss + reward_loss)

            total_loss += loss.item()
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
    num_epochs,
    device,
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_dataloader, optimizer, device)
        val_metrics = evaluate(model, val_dataloader, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "best_model.pth")
            print("  New best model saved!")

        if (epoch + 1) % 10 == 0:  # Test every 10 epochs
            model.load_state_dict(torch.load("best_model.pth"))
            test_metrics = evaluate(model, test_dataloader, device)
            print(f"Test Loss: {test_metrics['loss']:.4f}")


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    n_embed=256,
    state_shape=(14, 8, 8),
    action_shape=(73, 8, 8),
    dropout=0.1,
    device=device,
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Get dataloaders
train_dataloader, val_dataloader, test_dataloader = get_chess_dataloader(
    trajectory_length=10, num_games=10000
)

# Train the model
train_decision_transformer(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    num_epochs=100,
    device=device,
)
