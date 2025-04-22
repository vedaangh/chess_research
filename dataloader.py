import chess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from chess import engine
import chess.engine
import chess.pgn
from datetime import date
import os
import pickle
import torch.nn.functional as F
from tqdm import tqdm


class ChessDataset(Dataset):
    def __init__(
        self,
        num_games,
        max_moves,
        trajectory_length,
        engine_path,
        save_path=None,
        load_path=None,
        generate_new=False,
        skill_level=20,
        time_limit=0.1,
    ):
        self.num_games = num_games
        self.max_moves = max_moves
        self.trajectory_length = trajectory_length
        self.engine_path = engine_path
        self.save_path = save_path
        self.skill_level = skill_level
        self.time_limit = time_limit
        self.data = []

        board = chess.Board()
        self.state_shape = self.encode_board(board).shape
        dummy_move = next(iter(board.legal_moves), None)
        if dummy_move:
            self.action_shape = self.encode_move(dummy_move, board).shape
        else:
            self.action_shape = (73, 8, 8)

        processed_data_path = (
            load_path.replace(".pkl", "_processed.pkl") if load_path else None
        )
        raw_data_path = load_path

        if (
            processed_data_path
            and os.path.exists(processed_data_path)
            and not generate_new
        ):
            print(f"Loading processed dataset from {processed_data_path}")
            with open(processed_data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            raw_games = None
            if raw_data_path and os.path.exists(raw_data_path) and not generate_new:
                print(f"Loading raw game data from {raw_data_path}")
                with open(raw_data_path, "rb") as f:
                    raw_games = pickle.load(f)
            else:
                print(f"Generating {num_games} new games...")
                raw_games = self.generate_raw_chess_data(num_games)
                if self.save_path:
                    raw_save_path = self.save_path
                    print(f"Saving raw game data to {raw_save_path}")
                    with open(raw_save_path, "wb") as f:
                        pickle.dump(raw_games, f)

            print("Processing raw games into trajectories...")
            self.data = self._process_raw_games(raw_games)

            if self.save_path:
                processed_save_path = self.save_path.replace(".pkl", "_processed.pkl")
                print(f"Saving processed dataset to {processed_save_path}")
                with open(processed_save_path, "wb") as f:
                    pickle.dump(self.data, f)

        if not self.data:
            print("Warning: No data loaded or generated.")

    def generate_game(self):
        game_history = []
        try:
            with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
                engine.configure({"Skill Level": self.skill_level})
                board = chess.Board()

                for move_number in range(self.max_moves):
                    if board.is_game_over():
                        break

                    result = engine.play(
                        board, chess.engine.Limit(time=self.time_limit)
                    )
                    move = result.move

                    state = self.encode_board(board)
                    action = self.encode_move(move, board)
                    board_fen = board.fen()

                    game_history.append((state, action, board_fen))

                    board.push(move)

            return game_history

        except Exception as e:
            print(f"Error generating game: {e}")
            try:
                if "engine" in locals() and engine.is_alive():
                    engine.quit()
            except Exception as eq:
                print(f"Error quitting engine after exception: {eq}")
            return None

    def generate_raw_chess_data(self, num_games):
        raw_games_data = []
        for i in tqdm(range(num_games), desc="Generating Games"):
            game_data = self.generate_game()
            if game_data:
                raw_games_data.append(game_data)
        return raw_games_data

    def _process_raw_games(self, raw_games):
        processed_trajectories = []
        for game_history in tqdm(raw_games, desc="Processing Games"):
            if not game_history:
                continue

            states, actions, fens = zip(*game_history)
            states = list(states)
            actions = list(actions)

            final_fen = chess.Board(fens[-1]).fen()
            final_board_after_move = chess.Board(final_fen)
            final_board_after_move.push(chess.Move.from_uci(actions[-1]["uci"]))
            game_result = self.get_game_result(final_board_after_move)

            game_len = len(states)
            returns_to_go = np.zeros(game_len)
            current_rtg = game_result * (
                1 if final_board_after_move.turn == chess.BLACK else -1
            )
            for t in reversed(range(game_len)):
                returns_to_go[t] = current_rtg
                current_rtg = returns_to_go[t]

            returns_to_go.fill(game_result)

            for t in range(game_len - self.trajectory_length + 1):
                traj_states = states[t : t + self.trajectory_length]
                traj_actions = actions[t : t + self.trajectory_length]
                traj_rtgs = returns_to_go[t : t + self.trajectory_length]
                traj_timesteps = np.arange(t, t + self.trajectory_length)

                processed_trajectories.append(
                    {
                        "returns_to_go": np.array(traj_rtgs, dtype=np.float32).reshape(
                            -1, 1
                        ),
                        "states": np.array(traj_states, dtype=np.float32),
                        "actions": np.array(traj_actions, dtype=np.float32),
                        "timesteps": np.array(traj_timesteps, dtype=np.int64).reshape(
                            -1, 1
                        ),
                    }
                )
        return processed_trajectories

    def generate_example_game(self, max_moves=100, pgn_file="example_game.pgn"):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game

        white_skill = np.random.randint(1, 21)
        black_skill = np.random.randint(1, 21)

        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            for move_number in range(max_moves):
                if board.is_game_over():
                    break
                current_skill = (
                    white_skill if board.turn == chess.WHITE else black_skill
                )
                engine.configure({"Skill Level": current_skill})
                result = engine.play(board, chess.engine.Limit(time=self.time_limit))
                move = result.move
                node = node.add_variation(move)
                board.push(move)

        game.headers["Event"] = "Example Game"
        game.headers["Site"] = "Generated by ChessDataset"
        game.headers["Date"] = date.today().isoformat()
        game.headers["Round"] = "1"
        game.headers["White"] = f"Engine (Skill Level: {white_skill})"
        game.headers["Black"] = f"Engine (Skill Level: {black_skill})"
        game.headers["Result"] = board.result()

        with open(pgn_file, "w") as f:
            print(game, file=f, end="\n\n")

        print(f"Example game saved to {pgn_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "returns_to_go": torch.tensor(item["returns_to_go"], dtype=torch.float32),
            "states": torch.tensor(item["states"], dtype=torch.float32),
            "actions": torch.tensor(item["actions"], dtype=torch.float32),
            "timesteps": torch.tensor(item["timesteps"], dtype=torch.long),
        }

    def encode_board(self, board):
        state = np.zeros((14, 8, 8), dtype=np.float32)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                plane = (piece.piece_type - 1) * 2 + (
                    0 if piece.color == chess.WHITE else 1
                )
                state[plane, sq // 8, sq % 8] = 1
        if board.ep_square:
            state[12, board.ep_square // 8, board.ep_square % 8] = 1
        if board.turn == chess.WHITE:
            state[13, :, :] = 1

        legal_moves_planes = np.zeros((73, 8, 8), dtype=np.float32)
        if not board.is_game_over():
            try:
                for move in board.legal_moves:
                    legal_moves_planes += self._move_to_plane(move)
            except Exception as e:
                print(f"Error encoding legal moves for FEN {board.fen()}: {e}")

        state = np.concatenate([state, legal_moves_planes], axis=0)
        return state

    def _move_to_plane(self, move):
        action_planes = np.zeros(self.action_shape, dtype=np.float32)
        from_square = move.from_square
        to_square = move.to_square
        from_rank, from_file = divmod(from_square, 8)
        to_rank, to_file = divmod(to_square, 8)

        plane_index = -1
        if move.promotion is None:
            delta_rank = to_rank - from_rank
            delta_file = to_file - from_file
            distance = max(abs(delta_rank), abs(delta_file)) - 1

            if delta_rank == 0 and delta_file != 0:
                direction_offset = 0 if delta_file > 0 else 1
                plane_type = 0
            elif delta_file == 0 and delta_rank != 0:
                plane_type = 1
            elif abs(delta_rank) == abs(delta_file):
                plane_type = 2
            else:
                plane_type = 4

            if 0 <= distance < 7:
                plane_index = plane_type * 7 + distance
            else:
                print(f"Warning: Unexpected distance {distance} for move {move.uci()}")

        else:
            promo_type = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT].index(
                move.promotion
            )

            if to_file == from_file:
                capture_type = 0
            elif (board.turn == chess.WHITE and to_file > from_file) or (
                board.turn == chess.BLACK and to_file < from_file
            ):
                capture_type = 1
            else:
                capture_type = 2

            base_plane = [35, 44, 53, 62][promo_type]
            plane_index = base_plane + capture_type

        if plane_index != -1 and 0 <= plane_index < self.action_shape[0]:
            action_planes[plane_index, from_rank, from_file] = 1
        else:
            print(
                f"Warning: Calculated plane index {plane_index} out of bounds for move {move.uci()}"
            )

        return action_planes

    def encode_move(self, move, board):
        return self._move_to_plane(move)

    def get_game_result(self, board):
        result = board.result(claim_draw=True)
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        elif result == "1/2-1/2":
            return 0
        else:
            if board.is_checkmate():
                return 1 if board.turn == chess.BLACK else -1
            elif (
                board.is_stalemate()
                or board.is_insufficient_material()
                or board.is_seventyfive_moves()
                or board.is_fivefold_repetition()
            ):
                return 0
            else:
                return 0

    def decode_action(self, action_planes, board):
        if not isinstance(action_planes, np.ndarray):
            action_planes = action_planes.cpu().numpy()

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        move_scores = {}
        max_score = -np.inf
        best_move = None

        for move in legal_moves:
            try:
                move_plane = self._move_to_plane(move)
                plane_idx, from_rank, from_file = np.unravel_index(
                    np.argmax(move_plane), move_plane.shape
                )

                score = action_planes[plane_idx, from_rank, from_file]
                move_scores[move.uci()] = score

                if score > max_score:
                    max_score = score
                    best_move = move
            except Exception as e:
                print(
                    f"Error decoding/scoring move {move.uci()} on board {board.fen()}: {e}"
                )
                continue

        if best_move is None and legal_moves:
            best_move = legal_moves[0]
        elif not legal_moves:
            return None

        return best_move


def get_chess_dataloader(
    num_games=1000,
    max_moves=150,
    trajectory_length=20,
    batch_size=64,
    engine_path=None,
    save_path="chess_dataset_raw.pkl",
    load_path="chess_dataset_raw.pkl",
    generate_new=False,
    skill_level=20,
    time_limit=0.1,
    num_workers=4,
):
    if engine_path is None:
        raise ValueError("engine_path must be provided.")

    dataset = ChessDataset(
        num_games=num_games,
        max_moves=max_moves,
        trajectory_length=trajectory_length,
        engine_path=engine_path,
        save_path=save_path,
        load_path=load_path,
        generate_new=generate_new,
        skill_level=skill_level,
        time_limit=time_limit,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty after initialization. Check paths and generation."
        )

    total_size = len(dataset)
    if total_size < 3:
        raise ValueError(
            f"Dataset size ({total_size}) is too small for train/val/test split."
        )

    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = max(1, total_size - train_size - val_size)
    train_size = total_size - val_size - test_size

    if train_size + val_size + test_size > total_size:
        val_size -= 1
        if train_size + val_size + test_size > total_size:
            train_size -= 1

    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            "Calculated split results in zero size for one or more datasets."
        )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        dataset.state_shape,
        dataset.action_shape,
    )
