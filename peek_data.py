import pandas as pd

# Stream the CSV file in chunks (so we never load the entire file at once)
chunk_size = 100_000
csv_iter = pd.read_csv("data/lichess_db_standard_rated_2019-12.csv",
                       chunksize=chunk_size,
                       iterator=True)

# Grab the first chunk and show its head
first_chunk = next(csv_iter)
print("First few rows:")
print(first_chunk.head()["white_active"])
print(first_chunk.columns)
print()
