import chess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from chess import engine
import chess.engine

class ChessDataset(Dataset):
    def __init__(self, num_games, max_moves, trajectory_length, engine_path):
        self.trajectory_length = trajectory_length
        self.engine_path = engine_path
        self.data = self.generate_chess_data(num_games, max_moves)

    def generate_chess_data(self, num_games, max_moves):
        data = []
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            for _ in range(num_games):
                board = chess.Board()
                game_history = []
                for move_number in range(max_moves):
                    if board.is_game_over():
                        break
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                    state = self.encode_board(board)
                    action = self.encode_move(move, board)
                    game_history.append((state, action))
                    board.push(move)
                
                result = self.get_game_result(board)
                game_length = len(game_history)
                
                # Determine which player's perspective to use
                use_black_perspective = result == 1  # Black wins
                
                for start_idx in range(0, game_length - self.trajectory_length):
                    trajectory = game_history[start_idx:start_idx + self.trajectory_length + 1]
                    returns_to_go = [result * (0.99 ** (game_length - i - 1)) for i in range(start_idx, start_idx + self.trajectory_length + 1)]
                    
                    states = [t[0] for t in trajectory]
                    actions = [t[1] for t in trajectory]
                    timesteps = list(range(start_idx, start_idx + self.trajectory_length + 1))
                    
                    # Flip the perspective if using black's perspective
                    if use_black_perspective:
                        states = [self.flip_perspective(s) for s in states]
                        actions = [self.flip_perspective(a) for a in actions]
                        returns_to_go = [-r for r in returns_to_go]  # Negate returns for black's perspective
                    
                    data.append({
                        'returns_to_go': returns_to_go,
                        'states': states,
                        'actions': actions,
                        'timesteps': timesteps
                    })
    
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'returns_to_go': torch.tensor(item['returns_to_go'], dtype=torch.float32).unsqueeze(-1),
            'states': torch.tensor(np.stack(item['states']), dtype=torch.float32),
            'actions': torch.tensor(np.stack(item['actions']), dtype=torch.float32),
            'timesteps': torch.tensor(item['timesteps'], dtype=torch.long).unsqueeze(-1),
        }

    def encode_board(self, board):
        # 12 planes for each piece type and color, 1 plane for en passant, 1 plane for color to move
        state = np.zeros((14, 8, 8), dtype=np.float32)
        
        # Encode piece positions
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                state[color * 6 + piece_type, i // 8, i % 8] = 1
        
        # Encode en passant square
        if board.ep_square:
            state[12, board.ep_square // 8, board.ep_square % 8] = 1
        
        # Encode color to move
        state[13, :, :] = int(board.turn)
        
        return state

    def encode_move(self, move, board):
        # Create 73 planes to represent different move types
        action_planes = np.zeros((73, 8, 8), dtype=np.float32)
        
        from_square = move.from_square
        to_square = move.to_square
        from_rank, from_file = divmod(from_square, 8)
        to_rank, to_file = divmod(to_square, 8)
        
        # Determine the move type and set the corresponding plane
        if move.promotion is None:
            # Queen moves (including regular moves)
            delta_rank = to_rank - from_rank
            delta_file = to_file - from_file
            
            if delta_rank == 0:  # Horizontal move
                plane = 0
            elif delta_file == 0:  # Vertical move
                plane = 1
            elif abs(delta_rank) == abs(delta_file):  # Diagonal move
                plane = 2 if delta_rank * delta_file > 0 else 3
            else:  # Knight move
                plane = 4
            
            distance = max(abs(delta_rank), abs(delta_file)) - 1
            action_planes[plane * 7 + distance, from_rank, from_file] = 1
        else:
            # Promotions
            if move.promotion == chess.QUEEN:
                base_plane = 35
            elif move.promotion == chess.ROOK:
                base_plane = 44
            elif move.promotion == chess.BISHOP:
                base_plane = 53
            else:  # KNIGHT
                base_plane = 62
            
            if to_file == from_file:  # Straight promotion
                plane = base_plane
            elif to_file > from_file:  # Capture right
                plane = base_plane + 1
            else:  # Capture left
                plane = base_plane + 2
            
            action_planes[plane, from_rank, from_file] = 1
        
        return action_planes

    def get_game_result(self, board):
        if board.is_checkmate():
            return 1 if board.turn == chess.BLACK else -1
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0
        else:
            return 0  # In case of any other termination

    def flip_perspective(self, array):
        # Flip the board representation
        return np.flip(array, axis=(1, 2))

def get_chess_dataloader(num_games=1000, max_moves=100, trajectory_length=10, batch_size=32, engine_path="path/to/stockfish"):
    full_dataset = ChessDataset(num_games, max_moves, trajectory_length, engine_path)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# Example usage:
train_dataloader, val_dataloader, test_dataloader = get_chess_dataloader(trajectory_length=10, engine_path="/path/to/stockfish")
print(len(train_dataloader.dataset))
print(len(val_dataloader.dataset))
print(len(test_dataloader.dataset))
for batch in train_dataloader:
    returns_to_go, states, actions, timesteps = batch['returns_to_go'], batch['states'], batch['actions'], batch['timesteps']
    print(f"Returns-to-go shape: {returns_to_go.shape}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    
    # Separate the data as requested
    first_trajectory = {
        'returns_to_go': returns_to_go[:, :10],
        'states': states[:, :10],
        'actions': actions[:, :10],
        'timesteps': timesteps[:, :10]
    }
    
    second_trajectory = {
        'returns_to_go': returns_to_go[:, 1:],
        'states': states[:, 1:],
        'actions': actions[:, 1:],
        'timesteps': timesteps[:, 1:]
    }
    
    print("First trajectory (steps 0-9):")
    print(f"Returns-to-go shape: {first_trajectory['returns_to_go'].shape}")
    print(f"States shape: {first_trajectory['states'].shape}")
    print(f"Actions shape: {first_trajectory['actions'].shape}")
    print(f"Timesteps shape: {first_trajectory['timesteps'].shape}")
    
    print("\nSecond trajectory (steps 1-10):")
    print(f"Returns-to-go shape: {second_trajectory['returns_to_go'].shape}")
    print(f"States shape: {second_trajectory['states'].shape}")
    print(f"Actions shape: {second_trajectory['actions'].shape}")
    print(f"Timesteps shape: {second_trajectory['timesteps'].shape}")
    
    break  # Just print the first batch
