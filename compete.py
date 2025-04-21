import chess
import chess.engine
import chess.pgn
import torch
from model import Transformer
import argparse
from datetime import datetime
from dataloader import ChessDataset


def setup_model(model_path, device):
    # Initialize dataset without generating games
    dataset = ChessDataset(
        num_games=1,
        max_moves=100,
        trajectory_length=10,
        engine_path="/home/linuxbrew/.linuxbrew/bin/stockfish",
        generate_games=False,  # Don't generate games for prediction
    )

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
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def save_game_pgn(board, model_plays_white, game_num, skill_level, output_file):
    game = chess.pgn.Game()

    # Set headers
    game.headers["Event"] = f"Model vs Stockfish Game {game_num}"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = (
        "Model" if model_plays_white else f"Stockfish (Level {skill_level})"
    )
    game.headers["Black"] = (
        f"Stockfish (Level {skill_level})" if model_plays_white else "Model"
    )
    game.headers["Result"] = board.result()

    # Add moves
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)

    # Append to PGN file
    with open(output_file, "a") as f:
        print(game, file=f)
        print("\n", file=f)  # Add blank line between games

    return game


def play_game(
    model,
    engine_path,
    model_plays_white=True,
    skill_level=20,
    game_num=1,
    pgn_file="games.pgn",
    time_limit=0.1,
):
    try:
        # Initialize engine
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine.configure({"Skill Level": skill_level})
    except FileNotFoundError:
        print(f"Error: Stockfish not found at {engine_path}")
        print("Please install Stockfish and provide the correct path")
        return None, None, None

    board = chess.Board()
    game_moves = []

    try:
        rtgs = 10000.0 if model_plays_white else -10000.0
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                if model_plays_white:
                    # Model's turn
                    move = model.predict_move(
                        board=board,
                        rtg=rtgs,
                        timestep=len(game_moves),
                        context_window=10,
                    )
                    if move is None:  # No legal moves found
                        break
                    board.push(move)  # Add this line to actually make the move!
                    print(f"Model move: {move}")
                else:
                    # Stockfish's turn
                    result = engine.play(board, chess.engine.Limit(time=time_limit))
                    move = result.move
                    board.push(move)
                # Get the score of the board

            else:
                if model_plays_white:
                    # Stockfish's turn
                    result = engine.play(board, chess.engine.Limit(time=time_limit))
                    move = result.move
                    board.push(move)
                else:
                    # Model's turn
                    move = model.predict_move(
                        board=board,
                        rtg=rtgs,
                        timestep=len(game_moves),
                        context_window=10,
                    )
                    if move is None:  # No legal moves found
                        break
                    board.push(move)  # Add this line to actually make the move!
                    print(f"Model move: {move}")
            score = (
                engine.analyse(board, chess.engine.Limit(time=0.1))["score"]
                .white()
                .score(mate_score=10000)
            )
            rtgs -= score
            if move is None:
                break

            game_moves.append(move)
            board.push(move)

            # Print current position
            print("\n")
            print(board)
            print(f"Move {len(game_moves)}: {move}")

    finally:
        engine.quit()

    # Save game to PGN file
    game = save_game_pgn(board, model_plays_white, game_num, skill_level, pgn_file)

    return board.result(), game_moves, game


def main():
    parser = argparse.ArgumentParser(description="Play chess: Model vs Stockfish")
    parser.add_argument(
        "--model_path", default="best_model.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--engine_path",
        default="/home/linuxbrew/.linuxbrew/bin/stockfish",
        help="Path to Stockfish",
    )
    parser.add_argument(
        "--skill_levels",
        type=int,
        nargs="+",
        default=[i for i in range(1, 21)],  # Test against lower skill levels
        help="Stockfish skill levels to test against",
    )
    parser.add_argument(
        "--games_per_level",
        type=int,
        default=1,
        help="Number of games to play per skill level",
    )
    parser.add_argument(
        "--model_plays_white", action="store_true", help="Model plays as white"
    )
    parser.add_argument(
        "--pgn_file", default="model_vs_stockfish.pgn", help="Output PGN file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model(args.model_path, device)

    # Clear the PGN file at the start
    with open(args.pgn_file, "w") as f:
        f.write("")

    results = {}
    games = []  # Store all games
    total_wins = 0
    total_draws = 0
    total_games = 0

    for skill_level in args.skill_levels:
        print(f"\nTesting against Stockfish skill level {skill_level}")
        results[skill_level] = []

        for game_num in range(args.games_per_level):
            print(f"\nPlaying game {game_num + 1}/{args.games_per_level}")
            result, moves, game = play_game(
                model=model,
                engine_path=args.engine_path,
                model_plays_white=args.model_plays_white,
                skill_level=skill_level,
                game_num=f"skill_{skill_level}_game_{game_num + 1}",
                pgn_file=args.pgn_file,
            )
            results[skill_level].append(result)
            games.append(game)  # Store the game
            print(f"Game result: {result}")

        # Calculate statistics for this skill level
        wins = results[skill_level].count("1-0" if args.model_plays_white else "0-1")
        draws = results[skill_level].count("1/2-1/2")
        total_wins += wins
        total_draws += draws
        total_games += args.games_per_level

        print(f"\nResults vs Stockfish level {skill_level}:")
        print(f"Wins: {wins}/{args.games_per_level}")
        print(f"Draws: {draws}/{args.games_per_level}")
        print(f"Losses: {args.games_per_level - wins - draws}/{args.games_per_level}")

    # Save all games to PGN file
    with open(args.pgn_file, "w") as f:
        for game in games:
            if game is not None:
                print(game, file=f)
                print("\n", file=f)  # Add blank line between games

    # Print overall results
    print("\nOverall Results:")
    print(f"Total Wins: {total_wins}/{total_games} ({total_wins/total_games*100:.1f}%)")
    print(
        f"Total Draws: {total_draws}/{total_games} ({total_draws/total_games*100:.1f}%)"
    )
    print(
        f"Total Losses: {total_games-total_wins-total_draws}/{total_games} ({(total_games-total_wins-total_draws)/total_games*100:.1f}%)"
    )
    print(f"\nGames saved to {args.pgn_file}")


if __name__ == "__main__":
    main()
