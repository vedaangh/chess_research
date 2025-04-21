import $ from 'jquery';
import * as ChessJS from 'chess.js';

// Define Chess type since the typings are not perfect
const Chess = typeof ChessJS === 'function' ? ChessJS : ChessJS.Chess;

// Define Square type from chess.js
type Square = 'a8' | 'b8' | 'c8' | 'd8' | 'e8' | 'f8' | 'g8' | 'h8' |
             'a7' | 'b7' | 'c7' | 'd7' | 'e7' | 'f7' | 'g7' | 'h7' |
             'a6' | 'b6' | 'c6' | 'd6' | 'e6' | 'f6' | 'g6' | 'h6' |
             'a5' | 'b5' | 'c5' | 'd5' | 'e5' | 'f5' | 'g5' | 'h5' |
             'a4' | 'b4' | 'c4' | 'd4' | 'e4' | 'f4' | 'g4' | 'h4' |
             'a3' | 'b3' | 'c3' | 'd3' | 'e3' | 'f3' | 'g3' | 'h3' |
             'a2' | 'b2' | 'c2' | 'd2' | 'e2' | 'f2' | 'g2' | 'h2' |
             'a1' | 'b1' | 'c1' | 'd1' | 'e1' | 'f1' | 'g1' | 'h1';

// Interfaces
interface BoardState {
  fen: string;
  legal_moves: string[];
  is_game_over: boolean;
  result?: string;
  ai_move?: string;
}

interface AIMoveResponse extends BoardState {
  ai_type?: string;                // Added to track which AI is being used
  move_probability?: number;       // Probability of the chosen move
  highest_prob_move?: string;      // Move with highest probability overall
  highest_prob_value?: number;     // Value of the highest probability
}

// Game state
let board: any = null;
let game = new Chess();
let moveHistory: string[] = [];
let currentAI: string = 'Unknown';

// Initialize chessboard
async function initBoard() {
  // Get initial board state from server
  const response = await fetch('/api/board');
  const state: BoardState = await response.json();
  
  // Configure chessboard
  const config = {
    draggable: true,
    position: state.fen,
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: (window as any).pieceTheme || 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
  };
  
  // Initialize the board
  board = (window as any).Chessboard('board', config);
  
  updateStatus();
}

// Prevent dragging illegal pieces
function onDragStart(source: Square, piece: string) {
  // Don't allow moving pieces if the game is over
  if (game.game_over()) return false;
  
  // Only allow dragging of white pieces (player is white)
  if (piece.search(/^b/) !== -1) return false;
}

// After page loads, initialize the debug element
document.addEventListener('DOMContentLoaded', () => {
  // Set up a visible debug element
  const debugInfoElement = document.getElementById('debugInfo');
  if (debugInfoElement) {
    debugInfoElement.innerHTML = 'Debug: Waiting for a move...';
  }
});

// Handle move when a piece is dropped
async function onDrop(source: Square, target: Square) {
  // Try to make the move
  const move = game.move({
    from: source,
    to: target,
    promotion: 'q' // Always promote to queen for simplicity
  });
  
  // If illegal move, "snapback"
  if (move === null) return 'snapback';
  
  moveHistory.push(game.fen());

  // Update debug element
  const debugInfoElement = document.getElementById('debugInfo');
  if (debugInfoElement) {
    debugInfoElement.innerHTML = 'Debug: Processing move...';
  }

  // Set stockfish level if available
  const stockfishLevel = $('#stockfishLevel').val();
  
  // Make the AI move
  try {
    const response = await fetch('/api/move', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        move: `${source}${target}`,
        stockfish_level: stockfishLevel
      })
    });
    
    const result = await response.json();
    
    // Direct DOM manipulation to ensure the debug info is displayed
    if (debugInfoElement) {
      if (result.ai_move) {
        const aiUsed = result.ai_type || 'Unknown AI';
        let debugText = '';
        
        // First add highest probability move info if available
        if (result.highest_prob_move && result.highest_prob_value !== undefined) {
          const highestConfidence = (result.highest_prob_value * 100).toFixed(2);
          debugText += `Highest prob move: <b>${result.highest_prob_move}</b> (${highestConfidence}%)<br>`;
        }
        
        // Then add the chosen move info
        if (result.move_probability !== undefined && result.ai_type && result.ai_type.includes('Model')) {
          // Format for model with probability
          const confidencePercent = (result.move_probability * 100).toFixed(2);
          debugText += `Chosen move: <b>${result.ai_move}</b> | AI: <b>${aiUsed}</b> | Confidence: <b>${confidencePercent}%</b>`;
        } else {
          // Format for Stockfish or when probability is missing
          debugText += `Chosen move: <b>${result.ai_move}</b> | AI: <b>${aiUsed}</b>`;
        }
        
        debugInfoElement.innerHTML = debugText;
      } else {
        // No AI move was made
        debugInfoElement.innerHTML = 'No AI move was made';
      }
    }
    
    // Update the game with AI's move
    game.load(result.fen);
    
    // Update the board position
    board.position(result.fen);
    
    // If AI made a move, add to history
    if (result.ai_move) {
      moveHistory.push(game.fen());
      // Update AI info with probability if available
      updateAIInfo(result.ai_type || 'Stockfish', result.move_probability);
    }
    
    updateStatus(result);
  } catch (error) {
    console.error('Error making move:', error);
    
    // Show error in debug element
    if (debugInfoElement) {
      debugInfoElement.innerHTML = `Error: ${error}`;
    }
  }
}

// Update AI information display
function updateAIInfo(aiType: string, probability?: number) {
  const aiInfoElement = document.getElementById('aiInfo');
  if (!aiInfoElement) return;
  
  currentAI = aiType;
  
  // Debug info
  console.log('updateAIInfo called with:', { aiType, probability });
  
  // Display probability if available (for model moves)
  let displayText = `Currently using: ${aiType}`;
  if (probability !== undefined && aiType.includes('Model')) {
    const confidencePercent = (probability * 100).toFixed(2);
    displayText += ` (confidence: ${confidencePercent}%)`;
    
    // Debug final text
    console.log('Final AI info text:', displayText);
  }
  
  aiInfoElement.innerHTML = displayText;
  
  // Change background color based on AI type
  if (aiType.includes('Model')) {
    aiInfoElement.style.backgroundColor = '#d4edda';  // Green for model
  } else if (aiType.includes('Stockfish')) {
    aiInfoElement.style.backgroundColor = '#f8d7da';  // Red for stockfish
  } else {
    aiInfoElement.style.backgroundColor = '#e8f4f8';  // Default blue
  }
}

// Update visuals after drop
function onSnapEnd() {
  board.position(game.fen());
}

// Update the game status
function updateStatus(state?: BoardState) {
  const statusElement = document.getElementById('status');
  if (!statusElement) return;
  
  let status = '';
  
  if (state?.is_game_over || game.game_over()) {
    status = 'Game over, ';
    if (state?.result === '1-0') {
      status += 'White wins!';
    } else if (state?.result === '0-1') {
      status += 'Black wins!';
    } else {
      status += 'Draw!';
    }
  } else {
    status = `It's your turn (white)`;
  }
  
  statusElement.innerHTML = `Game status: ${status}`;
}

// Reset the game
async function resetGame() {
  try {
    const response = await fetch('/api/reset', {
      method: 'POST'
    });
    
    const result: BoardState = await response.json();
    
    // Reset the game
    game = new Chess(result.fen);
    moveHistory = [];
    
    // Update the board
    board.position(result.fen);
    
    // Reset AI info
    updateAIInfo('Waiting for first move...');
    
    updateStatus();
  } catch (error) {
    console.error('Error resetting game:', error);
  }
}

// Undo the last move
function undoMove() {
  if (moveHistory.length >= 2) {
    // Remove AI's move and player's move
    moveHistory.pop();
    const previousPosition = moveHistory.pop();
    
    // Update game and board
    if (previousPosition) {
      game.load(previousPosition);
      board.position(game.fen());
      updateStatus();
    }
  }
}

// Set up button handlers and stockfish level change
$(document).ready(function() {
  $('#resetBtn').on('click', resetGame);
  $('#undoBtn').on('click', undoMove);
  
  // Handle stockfish level change
  $('#stockfishLevel').on('change', function(this: HTMLSelectElement) {
    const level = $(this).val();
    fetch('/api/set_stockfish_level', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ level: level })
    }).catch(error => {
      console.error('Error setting stockfish level:', error);
    });
  });
  
  // Initialize the board when document is ready
  initBoard();
}); 