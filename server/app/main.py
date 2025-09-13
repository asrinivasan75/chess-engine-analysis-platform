#!/usr/bin/env python3
"""
FastAPI Chess Engine Analysis Platform
Advanced chess analysis server with engine integration, real-time analysis, and WebSocket support
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
import subprocess
import time
import uuid
import logging
import chess
import chess.engine
import chess.pgn
from pathlib import Path
import numpy as np
from contextlib import asynccontextmanager
import httpx
from datetime import datetime
import io
import redis
from celery import Celery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models
class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="FEN string of the position to analyze")
    depth: int = Field(10, ge=1, le=30, description="Search depth (1-30)")
    movetime: Optional[int] = Field(None, ge=100, le=30000, description="Time per move in milliseconds")
    nodes: Optional[int] = Field(None, ge=1000, description="Maximum nodes to search")
    multipv: int = Field(1, ge=1, le=5, description="Number of principal variations to return")
    engine: str = Field("default", description="Engine to use for analysis")

class AnalysisResult(BaseModel):
    fen: str
    depth: int
    score: Union[int, str]  # Centipawns or "mate in X"
    pv: List[str]  # Principal variation in algebraic notation
    nodes: int
    time_ms: int
    nps: int  # Nodes per second
    multipv: Optional[List[Dict[str, Any]]] = None
    engine: str
    
class GameAnalysisRequest(BaseModel):
    pgn: str = Field(..., description="PGN of the game to analyze")
    depth: int = Field(15, ge=5, le=25, description="Analysis depth")
    threshold: int = Field(50, ge=10, le=200, description="Centipawn threshold for mistakes")
    
class PositionEvaluation(BaseModel):
    move_number: int
    fen: str
    evaluation: int  # Centipawns from white's perspective
    best_move: str
    played_move: Optional[str]
    classification: str  # "book", "best", "good", "inaccuracy", "mistake", "blunder"
    
class GameAnalysisResult(BaseModel):
    game_id: str
    white_accuracy: float
    black_accuracy: float
    positions: List[PositionEvaluation]
    summary: Dict[str, Any]

class EngineConfig(BaseModel):
    name: str
    path: str
    options: Dict[str, Any] = {}
    enabled: bool = True

class EngineInfo(BaseModel):
    name: str
    version: str
    author: str
    options: Dict[str, Any]
    supported_features: List[str]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

# Chess Engine Manager
class ChessEngineManager:
    def __init__(self):
        self.engines: Dict[str, EngineConfig] = {}
        self.active_engines: Dict[str, chess.engine.SimpleEngine] = {}
        self.engine_locks: Dict[str, asyncio.Lock] = {}
        self.initialize_default_engines()
    
    def initialize_default_engines(self):
        """Initialize default engine configurations"""
        # Add our custom engine
        custom_engine_path = Path("../engine/build/chess_engine")
        if custom_engine_path.exists():
            self.engines["custom"] = EngineConfig(
                name="AadiChessEngine",
                path=str(custom_engine_path.absolute()),
                options={"Hash": 128, "Threads": 1},
                enabled=True
            )
        
        # Add Stockfish if available
        stockfish_paths = [
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
            "stockfish"  # System PATH
        ]
        
        for path in stockfish_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.engines["stockfish"] = EngineConfig(
                        name="Stockfish",
                        path=path,
                        options={"Hash": 128, "Threads": 1},
                        enabled=True
                    )
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        # Set default engine
        if "stockfish" in self.engines:
            self.engines["default"] = self.engines["stockfish"]
        elif "custom" in self.engines:
            self.engines["default"] = self.engines["custom"]
    
    async def get_engine(self, engine_name: str = "default") -> chess.engine.SimpleEngine:
        """Get or create an engine instance"""
        if engine_name not in self.engines:
            raise HTTPException(status_code=400, detail=f"Engine '{engine_name}' not found")
        
        engine_config = self.engines[engine_name]
        if not engine_config.enabled:
            raise HTTPException(status_code=400, detail=f"Engine '{engine_name}' is disabled")
        
        # Create lock for this engine if it doesn't exist
        if engine_name not in self.engine_locks:
            self.engine_locks[engine_name] = asyncio.Lock()
        
        # Check if engine is already running
        if engine_name in self.active_engines:
            return self.active_engines[engine_name]
        
        async with self.engine_locks[engine_name]:
            # Double-check after acquiring lock
            if engine_name in self.active_engines:
                return self.active_engines[engine_name]
            
            try:
                # Start engine
                transport, engine = await chess.engine.popen_uci(engine_config.path)
                
                # Configure engine options
                for option, value in engine_config.options.items():
                    if option in engine.options:
                        await engine.configure({option: value})
                
                self.active_engines[engine_name] = engine
                logger.info(f"Started engine: {engine_config.name} ({engine_name})")
                
                return engine
                
            except Exception as e:
                logger.error(f"Failed to start engine {engine_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start engine: {e}")
    
    async def analyze_position(self, 
                             fen: str, 
                             engine_name: str = "default",
                             depth: Optional[int] = None,
                             time_limit: Optional[float] = None,
                             nodes: Optional[int] = None,
                             multipv: int = 1) -> AnalysisResult:
        """Analyze a chess position"""
        start_time = time.time()
        
        try:
            # Parse position
            board = chess.Board(fen)
            if not board.is_valid():
                raise HTTPException(status_code=400, detail="Invalid FEN position")
            
            # Get engine
            engine = await self.get_engine(engine_name)
            
            # Set up analysis limits
            limit = chess.engine.Limit()
            if depth:
                limit.depth = depth
            if time_limit:
                limit.time = time_limit
            if nodes:
                limit.nodes = nodes
            
            # MultiPV is handled automatically by python-chess
            
            # Analyze position
            info = await engine.analyse(board, limit, multipv=multipv)
            
            # Extract results
            if isinstance(info, list):
                # Multiple PV lines
                main_info = info[0]
                multipv_results = []
                
                for i, pv_info in enumerate(info):
                    score = pv_info.get("score")
                    pv = pv_info.get("pv", [])
                    
                    # Convert score
                    if score:
                        if score.is_mate():
                            score_str = f"mate {score.mate()}"
                            score_cp = 29900 if score.mate() > 0 else -29900
                        else:
                            score_cp = score.relative.score()
                            score_str = str(score_cp)
                    else:
                        score_cp = 0
                        score_str = "0"
                    
                    # Convert PV to algebraic notation
                    pv_moves = []
                    temp_board = board.copy()
                    for move in pv[:10]:  # Limit PV length
                        pv_moves.append(temp_board.san(move))
                        temp_board.push(move)
                    
                    multipv_results.append({
                        "rank": i + 1,
                        "score": score_str,
                        "score_cp": score_cp,
                        "pv": pv_moves,
                        "depth": pv_info.get("depth", depth or 0)
                    })
                
                # Use main line for primary result
                score_display = multipv_results[0]["score"]
                pv_moves = multipv_results[0]["pv"]
                
            else:
                # Single PV line
                main_info = info
                score = main_info.get("score")
                pv = main_info.get("pv", [])
                
                # Convert score
                if score:
                    if score.is_mate():
                        score_display = f"mate {score.mate()}"
                    else:
                        score_display = str(score.relative.score())
                else:
                    score_display = "0"
                
                # Convert PV to algebraic notation
                pv_moves = []
                temp_board = board.copy()
                for move in pv[:10]:  # Limit PV length
                    pv_moves.append(temp_board.san(move))
                    temp_board.push(move)
                
                multipv_results = None
            
            # Calculate timing and speed
            analysis_time = int((time.time() - start_time) * 1000)
            nodes_searched = main_info.get("nodes", 0)
            nps = int(nodes_searched / max(analysis_time / 1000, 0.001))
            
            return AnalysisResult(
                fen=fen,
                depth=main_info.get("depth", depth or 0),
                score=score_display,
                pv=pv_moves,
                nodes=nodes_searched,
                time_ms=analysis_time,
                nps=nps,
                multipv=multipv_results,
                engine=engine_name
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    
    async def cleanup(self):
        """Cleanup all engine instances"""
        for engine_name, engine in self.active_engines.items():
            try:
                await engine.quit()
                logger.info(f"Closed engine: {engine_name}")
            except Exception as e:
                logger.error(f"Error closing engine {engine_name}: {e}")
        
        self.active_engines.clear()

# Global instances
engine_manager = ChessEngineManager()
connection_manager = ConnectionManager()

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Chess Analysis Platform...")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await engine_manager.cleanup()

# FastAPI app
app = FastAPI(
    title="Chess Engine Analysis Platform",
    description="Advanced chess analysis with multiple engines, real-time analysis, and comprehensive game evaluation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Engine information
@app.get("/engines", response_model=Dict[str, EngineInfo], tags=["Engines"])
async def list_engines():
    """List available engines"""
    engines_info = {}
    
    for name, config in engine_manager.engines.items():
        try:
            engine = await engine_manager.get_engine(name)
            engines_info[name] = EngineInfo(
                name=config.name,
                version=getattr(engine.id, "version", "unknown"),
                author=getattr(engine.id, "author", "unknown"),
                options={opt: engine.options[opt].default for opt in engine.options},
                supported_features=list(engine.options.keys())
            )
        except Exception as e:
            logger.error(f"Error getting info for engine {name}: {e}")
            engines_info[name] = EngineInfo(
                name=config.name,
                version="unknown",
                author="unknown",
                options={},
                supported_features=[]
            )
    
    return engines_info

# Single position analysis
@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_position(request: AnalyzeRequest):
    """Analyze a single chess position"""
    try:
        # Determine time limit
        time_limit = None
        if request.movetime:
            time_limit = request.movetime / 1000.0  # Convert to seconds
        
        result = await engine_manager.analyze_position(
            fen=request.fen,
            engine_name=request.engine,
            depth=request.depth,
            time_limit=time_limit,
            nodes=request.nodes,
            multipv=request.multipv
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# Game analysis
@app.post("/analyze-game", response_model=GameAnalysisResult, tags=["Analysis"])
async def analyze_game(request: GameAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze a complete game from PGN"""
    game_id = str(uuid.uuid4())
    
    try:
        # Parse PGN
        pgn_io = io.StringIO(request.pgn)
        game = chess.pgn.read_game(pgn_io)
        
        if not game:
            raise HTTPException(status_code=400, detail="Invalid PGN")
        
        # Start background analysis
        background_tasks.add_task(
            analyze_game_background, 
            game_id, 
            game, 
            request.depth, 
            request.threshold
        )
        
        return {"game_id": game_id, "status": "started", "message": "Game analysis started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Game analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Game analysis failed: {e}")

async def analyze_game_background(game_id: str, game, depth: int, threshold: int):
    """Background task for game analysis"""
    try:
        logger.info(f"Starting game analysis {game_id}")
        
        positions = []
        board = game.board()
        move_number = 0
        
        # Analyze each position
        for move in game.mainline_moves():
            move_number += 1
            fen = board.fen()
            
            # Analyze current position
            result = await engine_manager.analyze_position(
                fen=fen,
                depth=depth,
                multipv=2  # Get top 2 moves for comparison
            )
            
            # Get the best move
            best_move = result.pv[0] if result.pv else None
            played_move = board.san(move)
            
            # Classify the move
            classification = classify_move(result, played_move, threshold)
            
            # Parse evaluation
            eval_cp = 0
            if isinstance(result.score, str) and "mate" in result.score:
                eval_cp = 29900 if "mate" in result.score and int(result.score.split()[1]) > 0 else -29900
            else:
                eval_cp = int(result.score)
            
            positions.append(PositionEvaluation(
                move_number=move_number,
                fen=fen,
                evaluation=eval_cp,
                best_move=best_move,
                played_move=played_move,
                classification=classification
            ))
            
            # Make the move
            board.push(move)
            
            # Notify progress via WebSocket
            progress_message = {
                "type": "analysis_progress",
                "game_id": game_id,
                "progress": move_number / len(list(game.mainline_moves())),
                "current_move": move_number
            }
            await connection_manager.broadcast(json.dumps(progress_message))
        
        # Calculate accuracy and summary
        white_accuracy, black_accuracy = calculate_accuracy(positions)
        summary = generate_game_summary(positions)
        
        # Store results (in a real app, you'd use a database)
        game_results[game_id] = GameAnalysisResult(
            game_id=game_id,
            white_accuracy=white_accuracy,
            black_accuracy=black_accuracy,
            positions=positions,
            summary=summary
        )
        
        # Notify completion
        completion_message = {
            "type": "analysis_complete",
            "game_id": game_id,
            "white_accuracy": white_accuracy,
            "black_accuracy": black_accuracy
        }
        await connection_manager.broadcast(json.dumps(completion_message))
        
        logger.info(f"Completed game analysis {game_id}")
        
    except Exception as e:
        logger.error(f"Error in game analysis {game_id}: {e}")
        error_message = {
            "type": "analysis_error",
            "game_id": game_id,
            "error": str(e)
        }
        await connection_manager.broadcast(json.dumps(error_message))

def classify_move(result, played_move, threshold):
    """Classify a move based on engine analysis"""
    # This is a simplified classification
    # In practice, you'd compare the evaluation loss
    if not result.multipv or len(result.multipv) < 2:
        return "good"
    
    best_score = result.multipv[0]["score_cp"]
    
    # Find the played move in the variations
    played_score = None
    for var in result.multipv:
        if var["pv"] and var["pv"][0] == played_move:
            played_score = var["score_cp"]
            break
    
    if played_score is None:
        # Move not in top variations, likely a mistake
        return "mistake"
    
    score_loss = abs(best_score - played_score)
    
    if score_loss <= 10:
        return "best"
    elif score_loss <= 25:
        return "good"
    elif score_loss <= 50:
        return "inaccuracy"
    elif score_loss <= 100:
        return "mistake"
    else:
        return "blunder"

def calculate_accuracy(positions):
    """Calculate accuracy percentages for both players"""
    white_moves = [pos for pos in positions if pos.move_number % 2 == 1]
    black_moves = [pos for pos in positions if pos.move_number % 2 == 0]
    
    def calc_player_accuracy(moves):
        if not moves:
            return 100.0
        
        good_moves = sum(1 for move in moves if move.classification in ["best", "good"])
        return (good_moves / len(moves)) * 100.0
    
    return calc_player_accuracy(white_moves), calc_player_accuracy(black_moves)

def generate_game_summary(positions):
    """Generate game analysis summary"""
    classifications = {}
    for pos in positions:
        cls = pos.classification
        classifications[cls] = classifications.get(cls, 0) + 1
    
    return {
        "total_moves": len(positions),
        "classifications": classifications,
        "average_eval": np.mean([pos.evaluation for pos in positions]),
        "eval_swings": len([pos for pos in positions if abs(pos.evaluation) > 200])
    }

# Game analysis results
game_results: Dict[str, GameAnalysisResult] = {}

@app.get("/analyze-game/{game_id}", response_model=GameAnalysisResult, tags=["Analysis"])
async def get_game_analysis(game_id: str):
    """Get game analysis results"""
    if game_id not in game_results:
        raise HTTPException(status_code=404, detail="Game analysis not found")
    
    return game_results[game_id]

# WebSocket for real-time analysis
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "analyze":
                # Real-time position analysis
                analysis_id = str(uuid.uuid4())
                fen = message["fen"]
                depth = message.get("depth", 15)
                
                # Send immediate acknowledgment
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "analysis_started",
                        "analysis_id": analysis_id,
                        "fen": fen
                    }),
                    websocket
                )
                
                try:
                    # Perform analysis
                    result = await engine_manager.analyze_position(
                        fen=fen,
                        depth=depth,
                        multipv=3
                    )
                    
                    # Send results
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "analysis_result",
                            "analysis_id": analysis_id,
                            "result": result.dict()
                        }),
                        websocket
                    )
                    
                except Exception as e:
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "analysis_error",
                            "analysis_id": analysis_id,
                            "error": str(e)
                        }),
                        websocket
                    )
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

# Static files (for serving the React frontend)
# app.mount("/", StaticFiles(directory="../web/build", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
