#!/usr/bin/env python3
"""
NNUE-style Chess Engine Training Pipeline
Advanced neural network training with self-play data generation and SPRT testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import chess
import chess.engine
import chess.pgn
from pathlib import Path
import json
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import subprocess
import random
from typing import List, Tuple, Dict, Optional
import threading
import queue
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessPosition:
    """Represents a chess position with features and evaluation"""
    def __init__(self, fen: str, score: float, result: float, ply: int = 0):
        self.fen = fen
        self.score = score  # Engine evaluation in centipawns
        self.result = result  # Game result from white's perspective (1.0, 0.5, 0.0)
        self.ply = ply
        self.board = chess.Board(fen)
        
    def extract_features(self) -> torch.Tensor:
        """Extract NNUE-style features from position"""
        # 768 = 64 squares * 12 piece types (6 pieces * 2 colors)
        features = torch.zeros(768, dtype=torch.float32)
        
        # King square indices for both sides
        white_king_sq = self.board.king(chess.WHITE)
        black_king_sq = self.board.king(chess.BLACK)
        
        if white_king_sq is None or black_king_sq is None:
            return features
        
        # Feature extraction based on HalfKP (Half King-Piece) representation
        piece_offset = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                # Calculate feature index based on piece, color, and king positions
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_idx = piece_offset[piece.piece_type] + color_offset
                feature_idx = square * 12 + piece_idx
                features[feature_idx] = 1.0
                
        return features

class NNUEModel(nn.Module):
    """NNUE-style neural network for chess position evaluation"""
    
    def __init__(self, input_size=768, hidden_size=512, output_size=1):
        super(NNUEModel, self).__init__()
        
        # Feature transformer (input layer)
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Accumulator network
        self.accumulator = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.feature_transformer(x)
        x = self.accumulator(x)
        return torch.tanh(x)  # Output in range [-1, 1]

class ChessDataset(Dataset):
    """Dataset for chess positions with evaluations and game results"""
    
    def __init__(self, positions: List[ChessPosition]):
        self.positions = positions
        logger.info(f"Loaded {len(positions)} positions for training")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx]
        features = pos.extract_features()
        
        # Normalize evaluation score to roughly [-1, 1] range
        normalized_score = np.tanh(pos.score / 400.0)
        
        # Combine evaluation and game result with weighted average
        # This helps balance between accurate evaluation and game outcomes
        eval_weight = 0.7
        result_weight = 0.3
        target = eval_weight * normalized_score + result_weight * (pos.result * 2 - 1)
        
        return features, torch.tensor(target, dtype=torch.float32)

class SelfPlayDataGenerator:
    """Generates training data through engine self-play"""
    
    def __init__(self, engine_path: str, time_per_move: float = 0.1):
        self.engine_path = engine_path
        self.time_per_move = time_per_move
    
    def generate_game_data(self, num_games: int = 100) -> List[ChessPosition]:
        """Generate training data from self-play games"""
        positions = []
        
        logger.info(f"Generating {num_games} self-play games...")
        
        for game_idx in tqdm(range(num_games), desc="Self-play games"):
            try:
                game_positions = self._play_single_game()
                positions.extend(game_positions)
            except Exception as e:
                logger.warning(f"Error in game {game_idx}: {e}")
                continue
                
        logger.info(f"Generated {len(positions)} positions from self-play")
        return positions
    
    def _play_single_game(self) -> List[ChessPosition]:
        """Play a single game and extract positions"""
        board = chess.Board()
        positions = []
        
        # Simple random opening book
        opening_moves = random.randint(2, 8)
        for _ in range(opening_moves):
            if board.legal_moves:
                move = random.choice(list(board.legal_moves))
                board.push(move)
        
        # Play game with engine evaluation
        while not board.is_game_over() and len(positions) < 200:
            # Get engine evaluation (simplified - in real implementation use actual engine)
            eval_score = self._evaluate_position(board)
            
            # Store position
            positions.append(ChessPosition(
                fen=board.fen(),
                score=eval_score,
                result=0.5,  # Will be updated with game result
                ply=len(board.move_stack)
            ))
            
            # Make a move (simplified - random legal move)
            if board.legal_moves:
                move = random.choice(list(board.legal_moves))
                board.push(move)
            else:
                break
        
        # Update positions with game result
        result = self._get_game_result(board)
        for pos in positions:
            pos.result = result
            
        return positions
    
    def _evaluate_position(self, board: chess.Board) -> float:
        """Simple position evaluation (placeholder for actual engine)"""
        # Material count
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        
        # Add some positional factors
        score += random.gauss(0, 50)  # Add noise
        
        return score if board.turn == chess.WHITE else -score
    
    def _get_game_result(self, board: chess.Board) -> float:
        """Get game result from white's perspective"""
        if board.is_checkmate():
            return 0.0 if board.turn == chess.WHITE else 1.0
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.5
        else:
            return 0.5  # Unfinished game

class EloTracker:
    """Track Elo ratings and perform SPRT testing"""
    
    def __init__(self, base_elo: float = 1500.0):
        self.ratings = {'current': base_elo, 'baseline': base_elo}
        self.game_results = []
        self.sprt_results = []
        
    def add_game_result(self, result: float, player: str = 'current'):
        """Add a game result (1.0 = win, 0.5 = draw, 0.0 = loss)"""
        self.game_results.append((result, player, time.time()))
        
        # Update Elo rating
        opponent_elo = self.ratings['baseline'] if player == 'current' else self.ratings['current']
        new_elo = self._calculate_new_elo(self.ratings[player], opponent_elo, result)
        self.ratings[player] = new_elo
        
        logger.info(f"Game result: {result} for {player}, new Elo: {new_elo:.1f}")
    
    def _calculate_new_elo(self, player_elo: float, opponent_elo: float, score: float, k: float = 32) -> float:
        """Calculate new Elo rating using standard formula"""
        expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
        return player_elo + k * (score - expected_score)
    
    def perform_sprt(self, elo0: float = 0, elo1: float = 5, alpha: float = 0.05, beta: float = 0.05) -> Optional[bool]:
        """Perform Sequential Probability Ratio Test"""
        if len(self.game_results) < 10:
            return None
            
        # Calculate likelihood ratios
        wins = sum(1 for result, _, _ in self.game_results[-100:] if result > 0.5)
        draws = sum(1 for result, _, _ in self.game_results[-100:] if result == 0.5)
        losses = sum(1 for result, _, _ in self.game_results[-100:] if result < 0.5)
        
        total_games = wins + draws + losses
        if total_games == 0:
            return None
            
        score_rate = (wins + 0.5 * draws) / total_games
        
        # SPRT bounds
        upper_bound = np.log((1 - beta) / alpha)
        lower_bound = np.log(beta / (1 - alpha))
        
        # Log likelihood ratio (simplified)
        llr = total_games * (score_rate * np.log(score_rate / 0.5) + (1 - score_rate) * np.log((1 - score_rate) / 0.5))
        
        if llr >= upper_bound:
            logger.info(f"SPRT: H1 accepted (improvement detected) - LLR: {llr:.3f}")
            return True
        elif llr <= lower_bound:
            logger.info(f"SPRT: H0 accepted (no improvement) - LLR: {llr:.3f}")
            return False
        else:
            logger.info(f"SPRT: Continue testing - LLR: {llr:.3f}, bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            return None

class NNUETrainer:
    """Main trainer class for NNUE model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = NNUEModel(
            input_size=config.get('input_size', 768),
            hidden_size=config.get('hidden_size', 512),
            output_size=config.get('output_size', 1)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/nnue_training'))
        self.elo_tracker = EloTracker()
        
        # Data
        self.train_loader = None
        self.val_loader = None
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def load_data(self, data_path: str = None, generate_self_play: bool = True):
        """Load or generate training data"""
        positions = []
        
        if data_path and Path(data_path).exists():
            logger.info(f"Loading data from {data_path}")
            # Load existing data (implement based on your format)
            pass
        
        if generate_self_play:
            logger.info("Generating self-play data...")
            generator = SelfPlayDataGenerator("./engine/build/chess_engine")
            positions.extend(generator.generate_game_data(self.config.get('self_play_games', 100)))
        
        if not positions:
            logger.error("No training data available!")
            return
        
        # Create dataset
        dataset = ChessDataset(positions)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 1024),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 1024),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Training data: {len(train_dataset)} positions")
        logger.info(f"Validation data: {len(val_dataset)} positions")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (features, targets) in enumerate(progress_bar):
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features).squeeze()
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Training/Loss', loss.item(), step)
                self.writer.add_scalar('Training/LR', self.scheduler.get_last_lr()[0], step)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in tqdm(self.val_loader, desc="Validation"):
                features, targets = features.to(self.device), targets.to(self.device)
                
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Log to tensorboard
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/MSE', mse, epoch)
        self.writer.add_scalar('Validation/MAE', mae, epoch)
        
        # Add histogram of predictions vs targets
        self.writer.add_histogram('Validation/Predictions', np.array(all_predictions), epoch)
        self.writer.add_histogram('Validation/Targets', np.array(all_targets), epoch)
        
        return {'val_loss': avg_loss, 'mse': mse, 'mae': mae}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_path.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, checkpoint_path / 'best_model.pt')
            logger.info(f"New best model saved at epoch {epoch}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting NNUE training...")
        
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(self.config.get('max_epochs', 100)):
            logger.info(f"Epoch {epoch + 1}/{self.config.get('max_epochs', 100)}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Check for improvement
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"MAE: {val_metrics['mae']:.4f}"
            )
        
        logger.info("Training completed!")
        self.writer.close()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='NNUE Chess Engine Training')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--self-play', action='store_true', help='Generate self-play data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'input_size': 768,
        'hidden_size': 512,
        'output_size': 1,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'max_epochs': args.epochs,
        'min_lr': 1e-6,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'patience': 10,
        'self_play_games': 100,
        'log_dir': f'{args.output_dir}/runs',
        'checkpoint_dir': f'{args.output_dir}/checkpoints'
    }
    
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create output directories
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = NNUETrainer(config)
    
    # Load/generate data
    trainer.load_data(args.data_path, args.self_play)
    
    # Start training
    trainer.train()
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
