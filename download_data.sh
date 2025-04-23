#!/usr/bin/env bash
set -e
pwd
# Define the URL and data directory
URL="https://csslab.cs.toronto.edu/data/chess/monthly/lichess_db_standard_rated_2019-12.csv.bz2"
DATA_DIR="./data"

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Move to the data directory
cd $DATA_DIR

# Download the dataset if it doesn't exist already
if [ ! -f "$(basename "$URL")" ]; then
    wget -c "$URL"
fi

# Unzip the dataset
bunzip2 -f "$(basename "$URL")" 