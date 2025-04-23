import pandas as pd
# Define the CSV path and chunk size
csv_path = "data/lichess_db_standard_rated_2019-12.csv"
chunk_size = 10_000  # Adjust chunk size as needed
num_games_to_load = 6

# Create a CSV iterator
csv_iter = pd.read_csv(csv_path,
                       chunksize=chunk_size,
                       iterator=True)

# List to hold dataframes for the target games and set to track seen game IDs
games_data = []
seen_game_ids = set()
target_game_ids = []
found_all_targets = False
last_target_game_id = None

print(f"Loading the first {num_games_to_load} games (handling chunk splits)...")

# Iterate through chunks
for chunk_num, chunk in enumerate(csv_iter):
    print(f"Processing chunk {chunk_num + 1}...")
    
    # Find unique game IDs in the current chunk
    unique_ids_in_chunk = chunk['game_id'].unique()

    # Identify game IDs we haven't seen before
    newly_seen_ids = set(unique_ids_in_chunk) - seen_game_ids

    # Add new unique IDs to our target list if we still need games
    if not found_all_targets:
        for game_id in unique_ids_in_chunk: # Iterate in order they appear
            if game_id not in seen_game_ids:
                if len(target_game_ids) < num_games_to_load:
                    target_game_ids.append(game_id)
                    print(f"  Added target game ID {len(target_game_ids)}/{num_games_to_load}: {game_id}")
                    if len(target_game_ids) == num_games_to_load:
                        found_all_targets = True
                        last_target_game_id = target_game_ids[-1]
                        print(f"  Found all {num_games_to_load} target game IDs. Last ID: {last_target_game_id}")
                else:
                     # Already found 6, no need to check further in this chunk for *new* IDs
                     break
        # Update the set of all seen IDs
        seen_game_ids.update(newly_seen_ids)

    # Filter the chunk to include only rows from the target games *found so far*
    if target_game_ids: # Only filter if we have identified at least one target game
        filtered_chunk = chunk[chunk['game_id'].isin(target_game_ids)]
        if not filtered_chunk.empty:
            games_data.append(filtered_chunk)
            print(f"  Collected {len(filtered_chunk)} rows for target games.")

    # Termination condition: Stop only after we've found all target games
    # AND the current chunk does *not* contain the last target game ID.
    if found_all_targets:
        if last_target_game_id not in unique_ids_in_chunk:
            print(f"  Last target game {last_target_game_id} not found in chunk {chunk_num + 1}. Assuming it ended earlier. Stopping.")
            break
        else:
            print(f"  Last target game {last_target_game_id} is present in chunk {chunk_num + 1}. Continuing to ensure completeness.")

# Concatenate the collected dataframes
if games_data:
    full_df = pd.concat(games_data, ignore_index=True)
    # Final filter to ensure we only have rows from the exact target games
    first_n_games_df = full_df[full_df['game_id'].isin(target_game_ids)].copy()
    # Optional: Sort by game_id and then move_ply to ensure correct order if chunks were processed out of order (unlikely with iterator but safe)
    first_n_games_df.sort_values(by=['game_id', 'move_ply'], inplace=True)

    print("\n--------------------------------------------------")
    print(f"Data for the first {num_games_to_load} games loaded successfully.")
    print("\nDataFrame head:")
    print(first_n_games_df.head())
    print("\nDataFrame tail:")
    print(first_n_games_df.tail())
    print("\nUnique game IDs loaded:")
    loaded_ids = first_n_games_df['game_id'].unique()
    print(loaded_ids)
    print(f"Number of unique games loaded: {len(loaded_ids)}")
    print(f"\nTotal rows loaded: {len(first_n_games_df)}")
    # Verification Check
    if len(loaded_ids) == num_games_to_load:
        print("Verification successful: Correct number of unique games loaded.")
    else:
        print(f"Verification FAILED: Expected {num_games_to_load} games, but loaded {len(loaded_ids)}.")

    # Save the first 6 games to a CSV file
    output_file = "data/first_6_games.csv"
    first_n_games_df.to_csv(output_file, index=False)
    print(f"\nFirst {num_games_to_load} games saved to {output_file}")
else:
    print("No data loaded. Check the CSV file and parameters.")
