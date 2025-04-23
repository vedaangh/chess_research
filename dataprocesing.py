import pandas as pd
import numpy as np

filepath = "data/first_6_games.csv"
df_all = pd.read_csv(
    filepath,
    usecols=["game_id", "move_ply", "cp", "white_active"],
)
print(df_all.head())

# scale hyperparameter
k = 400.0

# choose mapping: logistic or tanh
def phi(cp, mapping="logistic", white_active):
    if mapping == "logistic":
        return 1.0 / (1.0 + np.exp(-cp / k)) if cp is not None else white_active * 1.0
    elif mapping == "tanh":
        return np.tanh(cp / k) if cp is not None else 2 * white_active * 1.0 - 1.0
    else:
        raise ValueError(f"Unknown mapping: {mapping}")

# Calculate phi(cp) for all moves
df_all['phi_cp'] = phi(df_all['cp'], mapping, df_all['white_active'])

temp = df_all

# Calculate phi(cp_{i-1}) within each game, using phi(0) for the first move
df_all['phi_cp_prev'] = df_all.groupby('game_id')['phi_cp'].shift(1)
df_all.loc[df_all['move_ply'] == 0, 'phi_cp_prev'] = phi(0, mapping, True)

# Calculate per-ply reward: R_i = phi(cp_i) - phi(cp_{i-1})
df_all['reward'] = df_all['phi_cp'] - df_all['phi_cp_prev']

# Calculate returns-to-go (rtg): Sum of future rewards (reverse cumulative sum)
df_all['rtg'] = df_all.groupby('game_id')['reward'].transform(
    lambda rewards: rewards[::-1].cumsum()[::-1]
)

# Clean up temporary columns
df_all = df_all.drop(columns=['phi_cp', 'phi_cp_prev', 'reward'])

# Save the processed data to a new CSV file
output_file = filepath+'processed.csv'
df_all.to_csv(output_file, index=False)
print(f"Processed data saved to {output_file}")
