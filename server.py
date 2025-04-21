from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chess
import chess.engine
import os
import json
from dataloader import ChessDataset
import torch
import numpy as np
from model import Transformer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Game state
board = chess.Board()
dataset = ChessDataset(
    num_games=0,  # We're not generating games
    max_moves=100,
    trajectory_length=10,
    engine_path="/usr/games/stockfish",  # Adjust this path as needed
    generate_games=False,
)

# Load model
MODEL_PATH = "best_model.pth"  # Adjust this path to where your model weights are saved
model = None

try:
    model = Transformer(
        d_model=512,
        num_heads=8,
        num_layers=4,
        d_ff=2048,
        n_embed=512,
        state_shape=(87, 8, 8),
        action_shape=(73, 8, 8),
        dropout=0.1,
        device=device,
        dataset=dataset,
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Initialize Stockfish engine as fallback
engine_path = "/usr/games/stockfish"  # Adjust path as needed
stockfish_engine = None
stockfish_level = 5  # Default level

try:
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    stockfish_engine.configure({"Skill Level": stockfish_level})
except Exception as e:
    print(f"Warning: Could not initialize Stockfish engine: {e}")


class MoveRequest(BaseModel):
    move: str
    stockfish_level: int = None


class StockfishLevelRequest(BaseModel):
    level: int


@app.get("/api/board")
def get_board():
    return {
        "fen": board.fen(),
        "legal_moves": [move.uci() for move in board.legal_moves],
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    }


@app.post("/api/move")
def make_move(move_request: MoveRequest):
    global board, stockfish_level, stockfish_engine

    # Update stockfish level if provided
    if move_request.stockfish_level is not None and stockfish_engine:
        try:
            new_level = int(move_request.stockfish_level)
            if 1 <= new_level <= 20:
                stockfish_level = new_level
                stockfish_engine.configure({"Skill Level": stockfish_level})
                print(f"Stockfish level set to {stockfish_level}")
        except:
            pass

    try:
        # Parse and validate the move
        chess_move = chess.Move.from_uci(move_request.move)
        if chess_move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")

        # Make the player's move
        board.push(chess_move)

        # If game is over after player's move, return
        if board.is_game_over():
            return {
                "fen": board.fen(),
                "legal_moves": [],
                "is_game_over": True,
                "result": board.result(),
            }

        # Try to use ML model if available
        ai_move = None
        ai_type = "Unknown"
        move_probability = None
        highest_prob_move = None
        highest_prob_value = None

        # First attempt to use the model's predict_move method
        if model is not None:
            try:
                print("DEBUG: Using ML model to predict move")

                # Get action probabilities from the model
                from torch.nn import functional as F

                # Prepare inputs for model forward pass - similar to predict_move
                board_state = dataset.encode_board(board)
                states = torch.zeros((1, 1, *model.state_shape), device=device)
                states[0, 0] = torch.tensor(board_state, device=device)

                rtgs = torch.zeros((1, 1, 1), device=device)
                timesteps = torch.zeros((1, 1, 1), device=device)

                with torch.no_grad():
                    # Get model prediction
                    action_preds, _, _ = model(
                        states=states,
                        actions=None,
                        rtgs=rtgs,
                        timesteps=timesteps,
                    )

                    # Get probabilities for each action - normalized across all possible moves
                    action_probs = (
                        F.softmax(action_preds[0, 0].view(-1), dim=0).cpu().numpy()
                    )

                    # Find the highest probability move overall, even if not legal
                    highest_prob_idx = action_probs.argmax()
                    highest_prob_value = float(action_probs[highest_prob_idx])

                    # Extract move parameters from the index
                    # Index structure: plane_number * 7 + distance * 8 * 8 + from_rank * 8 + from_file
                    # For non-promotion moves: plane 0-4, distance 0-6
                    # For promotion moves: plane 35-71

                    if highest_prob_idx < 35 * 8 * 8:  # Regular move
                        plane = highest_prob_idx // (7 * 8 * 8)
                        remainder = highest_prob_idx % (7 * 8 * 8)
                        distance = remainder // (8 * 8)
                        from_pos = remainder % (8 * 8)
                        from_rank = from_pos // 8
                        from_file = from_pos % 8

                        # Calculate to_rank and to_file based on move type and distance
                        if plane == 0:  # Horizontal
                            to_rank = from_rank
                            to_file = (
                                from_file + (distance + 1)
                                if from_file + (distance + 1) < 8
                                else from_file - (distance + 1)
                            )
                        elif plane == 1:  # Vertical
                            to_rank = (
                                from_rank + (distance + 1)
                                if from_rank + (distance + 1) < 8
                                else from_rank - (distance + 1)
                            )
                            to_file = from_file
                        elif plane == 2:  # Diagonal (positive slope)
                            dir_mult = 1 if from_rank < 4 else -1
                            to_rank = from_rank + dir_mult * (distance + 1)
                            to_file = from_file + dir_mult * (distance + 1)
                        elif plane == 3:  # Diagonal (negative slope)
                            dir_mult = 1 if from_rank < 4 else -1
                            to_rank = from_rank + dir_mult * (distance + 1)
                            to_file = from_file - dir_mult * (distance + 1)
                        else:  # Knight
                            knight_moves = [
                                (1, 2),
                                (2, 1),
                                (2, -1),
                                (1, -2),
                                (-1, -2),
                                (-2, -1),
                                (-2, 1),
                                (-1, 2),
                            ]
                            move_idx = distance % len(knight_moves)
                            to_rank = from_rank + knight_moves[move_idx][0]
                            to_file = from_file + knight_moves[move_idx][1]

                        # Ensure coordinates are within bounds
                        to_rank = max(0, min(7, to_rank))
                        to_file = max(0, min(7, to_file))

                        from_square = from_rank * 8 + from_file
                        to_square = to_rank * 8 + to_file

                        try:
                            highest_prob_move = chess.Move(from_square, to_square)
                            highest_prob_move_str = highest_prob_move.uci()
                        except:
                            highest_prob_move_str = f"{chess.square_name(from_square)}{chess.square_name(to_square)}"
                    else:
                        # Promotion move logic is more complex, providing a placeholder
                        highest_prob_move_str = "promotion-move"

                    print(
                        f"DEBUG: Highest probability move (may not be legal): {highest_prob_move_str} ({highest_prob_value:.4f})"
                    )

                    # Use the regular predict_move to get the legal move
                    ai_move = model.predict_move(
                        board=board,
                        rtg=0.0,
                        timestep=len(board.move_stack),
                        context_window=10,
                    )

                    if ai_move:
                        # Find the probability for the selected move
                        from_square = ai_move.from_square
                        to_square = ai_move.to_square
                        from_rank, from_file = divmod(from_square, 8)
                        to_rank, to_file = divmod(to_square, 8)

                        print(f"DEBUG: Legal move from {from_square} to {to_square}")
                        print(
                            f"DEBUG: from_rank={from_rank}, from_file={from_file}, to_rank={to_rank}, to_file={to_file}"
                        )

                        if ai_move.promotion is None:
                            delta_rank = to_rank - from_rank
                            delta_file = to_file - from_file
                            distance = max(abs(delta_rank), abs(delta_file)) - 1

                            if delta_rank == 0:  # Horizontal
                                plane = 0
                            elif delta_file == 0:  # Vertical
                                plane = 1
                            elif abs(delta_rank) == abs(delta_file):  # Diagonal
                                plane = 2 if delta_rank * delta_file > 0 else 3
                            else:  # Knight
                                plane = 4

                            idx = (
                                plane * 7 + distance * 8 * 8 + from_rank * 8 + from_file
                            )
                            move_probability = float(action_probs[idx])
                            print(
                                f"DEBUG: Non-promotion move. plane={plane}, distance={distance}, idx={idx}"
                            )
                        else:
                            # Handle promotions
                            if ai_move.promotion == chess.QUEEN:
                                base_plane = 35
                            elif ai_move.promotion == chess.ROOK:
                                base_plane = 44
                            elif ai_move.promotion == chess.BISHOP:
                                base_plane = 53
                            else:  # KNIGHT
                                base_plane = 62

                            if to_file == from_file:  # Straight
                                plane = base_plane
                            elif to_file > from_file:  # Capture right
                                plane = base_plane + 1
                            else:  # Capture left
                                plane = base_plane + 2

                            idx = plane * 8 * 8 + from_rank * 8 + from_file
                            move_probability = float(action_probs[idx])
                            print(
                                f"DEBUG: Promotion move. base_plane={base_plane}, plane={plane}, idx={idx}"
                            )

                        print(
                            f"DEBUG: Legal move probability={move_probability}, type={type(move_probability)}"
                        )
                        print(
                            f"Model move: {ai_move} (confidence: {move_probability:.4f})"
                        )
                        ai_type = "Trained Model"
            except Exception as e:
                import traceback

                print(f"Error using ML model: {e}")
                print(traceback.format_exc())
                ai_move = None

        # Fallback to Stockfish if ML model fails or isn't available
        if ai_move is None and stockfish_engine is not None:
            try:
                print("DEBUG: Falling back to Stockfish")
                result = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
                ai_move = result.move
                print(f"Stockfish move (level {stockfish_level}): {ai_move}")
                ai_type = f"Stockfish (Level {stockfish_level})"
            except Exception as e:
                print(f"Error using Stockfish: {e}")

        # Make AI move if one was found
        if ai_move:
            board.push(ai_move)

        # Create response with proper float value for move_probability
        response_data = {
            "fen": board.fen(),
            "legal_moves": [move.uci() for move in board.legal_moves],
            "ai_move": ai_move.uci() if ai_move else None,
            "is_game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "ai_type": ai_type,
        }

        # Only add move_probability if it exists and is a valid number
        if move_probability is not None:
            response_data["move_probability"] = float(move_probability)
            print(
                f"DEBUG: Adding move_probability to response: {float(move_probability)}"
            )

        # Add highest probability move information
        if highest_prob_move_str is not None and highest_prob_value is not None:
            response_data["highest_prob_move"] = highest_prob_move_str
            response_data["highest_prob_value"] = float(highest_prob_value)
            print(
                f"DEBUG: Adding highest_prob_move to response: {highest_prob_move_str} ({highest_prob_value:.4f})"
            )

        # Debug complete response
        print(f"DEBUG: Final response: {json.dumps(response_data, default=str)}")

        return response_data
    except Exception as e:
        import traceback

        print(f"Error in make_move: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set_stockfish_level")
def set_stockfish_level(level_request: StockfishLevelRequest):
    global stockfish_level, stockfish_engine

    if stockfish_engine is None:
        raise HTTPException(status_code=400, detail="Stockfish engine not available")

    try:
        level = int(level_request.level)
        if 1 <= level <= 20:
            stockfish_level = level
            stockfish_engine.configure({"Skill Level": stockfish_level})
            return {"status": "success", "level": stockfish_level}
        else:
            raise HTTPException(
                status_code=400, detail="Level must be between 1 and 20"
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid level value")


@app.post("/api/reset")
def reset_game():
    global board
    board = chess.Board()
    return {
        "fen": board.fen(),
        "legal_moves": [move.uci() for move in board.legal_moves],
        "is_game_over": False,
    }


# Add static files
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


# Clean up when application stops
@app.on_event("shutdown")
def shutdown_event():
    if stockfish_engine:
        stockfish_engine.quit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
