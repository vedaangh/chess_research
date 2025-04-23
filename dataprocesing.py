import pandas as pd
import numpy as np

df_all = pd.read_csv(
    "data/lichess_db_standard_rated_2019-12.csv",
    usecols=["game_id", "move_ply", "cp"],
)

# scale hyperparameter
k = 400.0

# choose mapping: logistic or tanh
def phi(cp, mapping="logistic"):
    if mapping == "logistic":
        return 1.0 / (1.0 + np.exp(-cp / k))
    elif mapping == "tanh":
        return np.tanh(cp / k)
    else:
        raise ValueError(f"Unknown mapping: {mapping}")

# compute returns-to-go per game (bottom-up, preserve original order)
# choose mapping: "logistic" (default) or "tanh"
mapping = "logistic"
df_all["rtg"] = df_all.groupby("game_id")["cp"].transform(
    lambda cp_series: phi(cp_series, mapping)[::-1].cumsum()[::-1]
)
# subtract baseline phi(0) only at the first move of each game
df_all.loc[df_all["move_ply"] == 0, "rtg"] -= phi(0, mapping)

