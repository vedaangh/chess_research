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


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train Decision Transformer for Chess")
parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of attention heads"
)
parser.add_argument(
    "--num_layers", type=int, default=4, help="Number of transformer layers"
)
parser.add_argument(
    "--d_ff", type=int, default=2048, help="Dimension of feed-forward layer"
)
parser.add_argument(
    "--max_seq_length", type=int, default=20, help="Maximum trajectory length (L)"
)
parser.add_argument("--n_embed", type=int, default=512, help="Embedding dimension")
parser.add_argument(
    "--num_epochs", type=int, default=100, help="Number of training epochs"
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_games", type=int, default=1000, help="Number of games to generate/load"
)
parser.add_argument("--max_moves", type=int, default=150, help="Maximum moves per game")
parser.add_argument(
    "--engine_path",
    type=str,
    default="/home/linuxbrew/.linuxbrew/bin/stockfish",
    help="Path to Stockfish engine",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="chess_dataset_raw.pkl",
    help="Path to load/save raw game data",
)
parser.add_argument(
    "--generate_new_data", action="store_true", help="Force generation of new dataset"
)
parser.add_argument(
    "--save_model_path",
    type=str,
    default="best_model.pth",
    help="Path to save best model",
)
parser.add_argument(
    "--log_path", type=str, default="loss_log.csv", help="Path to save training log"
)
parser.add_argument(
    "--stability_weight",
    type=float,
    default=0.001,
    help="Weight for state/reward stability losses",
)
parser.add_argument(
    "--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping"
)
parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers for dataloader"
)

args = parser.parse_args()


def generate_mask(seq_length, device):
    total_len = seq_length * 3 - 1
    mask = torch.tril(torch.ones(total_len, total_len, device=device))
    return mask


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    max_seq_length,
    stability_weight,
    clip_grad_norm,
):
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    num_batches = 0
    last_grad_norm = 0

    for data in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        R, S, A, T = [
            d.to(device)
            for d in [
                data["returns_to_go"],
                data["states"],
                data["actions"],
                data["timesteps"],
            ]
        ]

        states_input = S
        actions_input = A[:, :-1]
        rtgs_input = R
        timesteps_input = T

        attn_mask = generate_mask(max_seq_length, device)
        attn_mask = attn_mask.unsqueeze(0).expand(S.shape[0], -1, -1)

        a_pred, s_pred, r_pred = model(
            states=states_input,
            actions=actions_input,
            rtgs=rtgs_input,
            timesteps=timesteps_input,
            mask=attn_mask,
        )

        a_target = A
        s_target = S[:, 1:]
        r_target = R[:, 1:]

        action_loss = F.cross_entropy(
            a_pred.reshape(-1, *a_pred.shape[2:]),
            torch.argmax(a_target.reshape(-1, *a_target.shape[2:]), dim=1),
        )
        state_loss = F.mse_loss(s_pred, s_target)
        reward_loss = F.mse_loss(r_pred, r_target)

        loss = action_loss + stability_weight * (state_loss + reward_loss)
        loss.backward()

        last_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_grad_norm
        )

        optimizer.step()

        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_state_loss += state_loss.item()
        total_reward_loss += reward_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_state_loss = total_state_loss / num_batches
    avg_reward_loss = total_reward_loss / num_batches

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "state_loss": avg_state_loss,
        "reward_loss": avg_reward_loss,
        "final_gradient_norm": (
            last_grad_norm.item() if torch.is_tensor(last_grad_norm) else last_grad_norm
        ),
    }


def evaluate(model, dataloader, device, max_seq_length, stability_weight):
    model.eval()
    total_loss = 0
    total_action_loss = 0
    total_state_loss = 0
    total_reward_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", leave=False):
            R, S, A, T = [
                d.to(device)
                for d in [
                    data["returns_to_go"],
                    data["states"],
                    data["actions"],
                    data["timesteps"],
                ]
            ]

            states_input = S
            actions_input = A[:, :-1]
            rtgs_input = R
            timesteps_input = T

            attn_mask = generate_mask(max_seq_length, device)
            attn_mask = attn_mask.unsqueeze(0).expand(S.shape[0], -1, -1)

            a_pred, s_pred, r_pred = model(
                states=states_input,
                actions=actions_input,
                rtgs=rtgs_input,
                timesteps=timesteps_input,
                mask=attn_mask,
            )

            a_target = A
            s_target = S[:, 1:]
            r_target = R[:, 1:]

            action_loss = F.cross_entropy(
                a_pred.reshape(-1, *a_pred.shape[2:]),
                torch.argmax(a_target.reshape(-1, *a_target.shape[2:]), dim=1),
            )
            state_loss = F.mse_loss(s_pred, s_target)
            reward_loss = F.mse_loss(r_pred, r_target)

            loss = action_loss + stability_weight * (state_loss + reward_loss)

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_state_loss = total_state_loss / num_batches
    avg_reward_loss = total_reward_loss / num_batches

    return {
        "loss": avg_loss,
        "action_loss": avg_action_loss,
        "state_loss": avg_state_loss,
        "reward_loss": avg_reward_loss,
    }


def train_decision_transformer(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    max_seq_length,
    stability_weight,
    clip_grad_norm,
    log_path,
    save_model_path,
):
    best_val_loss = float("inf")

    with open(log_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "epoch",
                "type",
                "loss",
                "action_loss",
                "state_loss",
                "reward_loss",
                "lr",
                "grad_norm",
            ]
        )

        for epoch in range(num_epochs):
            train_metrics = train_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                max_seq_length,
                stability_weight,
                clip_grad_norm,
            )
            val_metrics = evaluate(
                model, val_dataloader, device, max_seq_length, stability_weight
            )

            lr = optimizer.param_groups[0]["lr"]
            grad_norm = train_metrics["final_gradient_norm"]

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f}, Action Loss: {train_metrics['action_loss']:.4f}, State Loss: {train_metrics['state_loss']:.4f}, Reward Loss: {train_metrics['reward_loss']:.4f}"
            )
            print(
                f"  Val   Loss: {val_metrics['loss']:.4f}, Action Loss: {val_metrics['action_loss']:.4f}, State Loss: {val_metrics['state_loss']:.4f}, Reward Loss: {val_metrics['reward_loss']:.4f}"
            )
            print(f"  LR: {lr:.6f}, Grad Norm: {grad_norm:.4f}")

            csvwriter.writerow(
                [
                    epoch + 1,
                    "train",
                    f"{train_metrics['loss']:.4f}",
                    f"{train_metrics['action_loss']:.4f}",
                    f"{train_metrics['state_loss']:.4f}",
                    f"{train_metrics['reward_loss']:.4f}",
                    f"{lr:.6f}",
                    f"{grad_norm:.4f}",
                ]
            )
            csvwriter.writerow(
                [
                    epoch + 1,
                    "validation",
                    f"{val_metrics['loss']:.4f}",
                    f"{val_metrics['action_loss']:.4f}",
                    f"{val_metrics['state_loss']:.4f}",
                    f"{val_metrics['reward_loss']:.4f}",
                    f"{lr:.6f}",
                    "N/A",
                ]
            )
            csvfile.flush()

            scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), save_model_path)
                print(f"  New best model saved to {save_model_path}!")

            if (epoch + 1) % 10 == 0:
                print("Running test evaluation...")
                test_metrics = evaluate(
                    model, test_dataloader, device, max_seq_length, stability_weight
                )
                print(
                    f"  Test  Loss: {test_metrics['loss']:.4f}, Action Loss: {test_metrics['action_loss']:.4f}, State Loss: {test_metrics['state_loss']:.4f}, Reward Loss: {test_metrics['reward_loss']:.4f}"
                )
                csvwriter.writerow(
                    [
                        epoch + 1,
                        "test",
                        f"{test_metrics['loss']:.4f}",
                        f"{test_metrics['action_loss']:.4f}",
                        f"{test_metrics['state_loss']:.4f}",
                        f"{test_metrics['reward_loss']:.4f}",
                        f"{lr:.6f}",
                        "N/A",
                    ]
                )
                csvfile.flush()


def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing Dataloaders...")
    train_dataloader, val_dataloader, test_dataloader, state_shape, action_shape = (
        get_chess_dataloader(
            num_games=args.num_games,
            max_moves=args.max_moves,
            trajectory_length=args.max_seq_length,
            batch_size=args.batch_size,
            engine_path=args.engine_path,
            save_path=args.dataset_path,
            load_path=args.dataset_path,
            generate_new=args.generate_new_data,
            num_workers=args.num_workers,
        )
    )
    print(f"State shape: {state_shape}, Action shape: {action_shape}")

    print("Initializing Model...")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        n_embed=args.n_embed,
        state_shape=state_shape,
        action_shape=action_shape,
        max_seq_length=args.max_seq_length,
        dropout=0.1,
        device=device,
    )
    model.to(device)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    print("Starting Training...")
    train_decision_transformer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        stability_weight=args.stability_weight,
        clip_grad_norm=args.clip_grad_norm,
        log_path=args.log_path,
        save_model_path=args.save_model_path,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
