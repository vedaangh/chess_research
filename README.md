# Chess Game vs AI

A web application that allows you to play chess against an AI. The backend uses FastAPI and Python-Chess, with a TypeScript/jQuery frontend displaying the chessboard.

## Prerequisites

- Python 3.7 or higher
- Node.js and npm
- Stockfish chess engine (optional, but recommended)

## Setup

### 1. Install Python dependencies

```bash
pip install fastapi uvicorn python-chess numpy torch
```

### 2. Install and build the frontend

```bash
cd frontend
npm install
npm run build
```

### 3. Adjust the Stockfish path

Open `server.py` and adjust the path to the Stockfish engine:

```python
engine_path = "/usr/games/stockfish"  # Change this to the path where Stockfish is installed
```

On Windows, this might be something like:
```python
engine_path = "C:\\path\\to\\stockfish.exe"
```

## Running the Application

1. Start the server:

```bash
python server.py
```

2. Open your browser and navigate to:

```
http://localhost:8000
```

## How to Play

- You play as white (the pieces at the bottom)
- Drag and drop your pieces to make moves
- The AI will automatically respond
- Use the "New Game" button to reset the board
- Use the "Undo Move" button to take back your last move (and the AI's response)

## Technical Details

- The server uses FastAPI to provide RESTful endpoints
- The AI uses either a machine learning model or falls back to Stockfish
- The frontend is built with TypeScript and jQuery
- The chessboard UI uses the chessboard.js library
- Chess logic is handled by the chess.js library 