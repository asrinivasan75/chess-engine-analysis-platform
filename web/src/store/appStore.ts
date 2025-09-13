import { create } from 'zustand'
import { Chess } from 'chess.js'

// Types
export interface AnalysisResult {
  fen: string
  depth: number
  score: string | number
  pv: string[]
  nodes: number
  time_ms: number
  nps: number
  multipv?: Array<{
    rank: number
    score: string
    score_cp: number
    pv: string[]
    depth: number
  }>
  engine: string
}

export interface GameAnalysis {
  gameId: string
  whiteAccuracy: number
  blackAccuracy: number
  positions: Array<{
    moveNumber: number
    fen: string
    evaluation: number
    bestMove: string
    playedMove?: string
    classification: string
  }>
  summary: {
    totalMoves: number
    classifications: Record<string, number>
    averageEval: number
    evalSwings: number
  }
}

export interface EngineInfo {
  name: string
  version: string
  author: string
  options: Record<string, any>
  supportedFeatures: string[]
}

export interface WebSocketMessage {
  type: 'analysis_started' | 'analysis_result' | 'analysis_error' | 'analysis_progress' | 'analysis_complete'
  analysisId?: string
  gameId?: string
  fen?: string
  result?: AnalysisResult
  error?: string
  progress?: number
  currentMove?: number
  whiteAccuracy?: number
  blackAccuracy?: number
}

// Store interfaces
interface ChessState {
  game: Chess
  position: string
  moveHistory: string[]
  currentMoveIndex: number
  isFlipped: boolean
}

interface AnalysisState {
  currentAnalysis: AnalysisResult | null
  isAnalyzing: boolean
  analysisHistory: AnalysisResult[]
  gameAnalysis: GameAnalysis | null
  isGameAnalyzing: boolean
  gameAnalysisProgress: number
}

interface EngineState {
  engines: Record<string, EngineInfo>
  selectedEngine: string
  engineStatus: Record<string, 'online' | 'offline'>
}

interface WebSocketState {
  isConnected: boolean
  connectionStatus: 'connected' | 'connecting' | 'disconnected'
  lastMessage: WebSocketMessage | null
  reconnectAttempts: number
}

interface UIState {
  sidebarOpen: boolean
  activeTab: string
  theme: 'light' | 'dark'
  boardTheme: string
  showCoordinates: boolean
  showMoveHints: boolean
  autoAnalyze: boolean
  analysisDepth: number
}

interface AppStore extends ChessState, AnalysisState, EngineState, WebSocketState, UIState {
  // Chess actions
  makeMove: (move: string) => boolean
  undoMove: () => void
  goToMove: (moveIndex: number) => void
  resetGame: () => void
  loadPosition: (fen: string) => void
  flipBoard: () => void
  
  // Analysis actions
  startAnalysis: (fen?: string, depth?: number) => void
  stopAnalysis: () => void
  setAnalysisResult: (result: AnalysisResult) => void
  clearAnalysisHistory: () => void
  startGameAnalysis: (pgn: string) => void
  setGameAnalysisResult: (result: GameAnalysis) => void
  setGameAnalysisProgress: (progress: number) => void
  
  // Engine actions
  setEngines: (engines: Record<string, EngineInfo>) => void
  selectEngine: (engineName: string) => void
  setEngineStatus: (engineName: string, status: 'online' | 'offline') => void
  
  // WebSocket actions
  setConnectionStatus: (status: 'connected' | 'connecting' | 'disconnected') => void
  setLastMessage: (message: WebSocketMessage) => void
  incrementReconnectAttempts: () => void
  resetReconnectAttempts: () => void
  
  // UI actions
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  setActiveTab: (tab: string) => void
  setTheme: (theme: 'light' | 'dark') => void
  setBoardTheme: (theme: string) => void
  setShowCoordinates: (show: boolean) => void
  setShowMoveHints: (show: boolean) => void
  setAutoAnalyze: (auto: boolean) => void
  setAnalysisDepth: (depth: number) => void
}

// Create store
export const useAppStore = create<AppStore>((set, get) => ({
  // Initial state
  game: new Chess(),
  position: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  moveHistory: [],
  currentMoveIndex: -1,
  isFlipped: false,
  
  currentAnalysis: null,
  isAnalyzing: false,
  analysisHistory: [],
  gameAnalysis: null,
  isGameAnalyzing: false,
  gameAnalysisProgress: 0,
  
  engines: {},
  selectedEngine: 'default',
  engineStatus: {},
  
  isConnected: false,
  connectionStatus: 'disconnected',
  lastMessage: null,
  reconnectAttempts: 0,
  
  sidebarOpen: true,
  activeTab: 'analysis',
  theme: 'dark',
  boardTheme: 'neo',
  showCoordinates: true,
  showMoveHints: true,
  autoAnalyze: true,
  analysisDepth: 15,
  
  // Chess actions
  makeMove: (move: string) => {
    const { game } = get()
    try {
      const moveObj = game.move(move)
      if (moveObj) {
        const newHistory = [...get().moveHistory, moveObj.san]
        set({
          position: game.fen(),
          moveHistory: newHistory,
          currentMoveIndex: newHistory.length - 1,
        })
        
        // Auto-analyze if enabled
        if (get().autoAnalyze && get().isConnected) {
          get().startAnalysis()
        }
        
        return true
      }
    } catch (error) {
      console.error('Invalid move:', error)
    }
    return false
  },
  
  undoMove: () => {
    const { game } = get()
    const move = game.undo()
    if (move) {
      const newHistory = get().moveHistory.slice(0, -1)
      set({
        position: game.fen(),
        moveHistory: newHistory,
        currentMoveIndex: newHistory.length - 1,
      })
    }
  },
  
  goToMove: (moveIndex: number) => {
    const { moveHistory } = get()
    const game = new Chess()
    
    for (let i = 0; i <= moveIndex && i < moveHistory.length; i++) {
      game.move(moveHistory[i])
    }
    
    set({
      game,
      position: game.fen(),
      currentMoveIndex: moveIndex,
    })
  },
  
  resetGame: () => {
    const game = new Chess()
    set({
      game,
      position: game.fen(),
      moveHistory: [],
      currentMoveIndex: -1,
      currentAnalysis: null,
      analysisHistory: [],
    })
  },
  
  loadPosition: (fen: string) => {
    try {
      const game = new Chess(fen)
      set({
        game,
        position: fen,
        moveHistory: [],
        currentMoveIndex: -1,
        currentAnalysis: null,
      })
      
      // Auto-analyze if enabled
      if (get().autoAnalyze && get().isConnected) {
        get().startAnalysis()
      }
    } catch (error) {
      console.error('Invalid FEN:', error)
    }
  },
  
  flipBoard: () => {
    set({ isFlipped: !get().isFlipped })
  },
  
  // Analysis actions
  startAnalysis: (fen?: string, depth?: number) => {
    const analysisDepth = depth || get().analysisDepth
    const currentFen = fen || get().position
    
    set({
      isAnalyzing: true,
      currentAnalysis: null,
    })
    
    // WebSocket message will be sent by the component
  },
  
  stopAnalysis: () => {
    set({
      isAnalyzing: false,
    })
  },
  
  setAnalysisResult: (result: AnalysisResult) => {
    set((state) => ({
      currentAnalysis: result,
      isAnalyzing: false,
      analysisHistory: [result, ...state.analysisHistory.slice(0, 9)], // Keep last 10
    }))
  },
  
  clearAnalysisHistory: () => {
    set({
      analysisHistory: [],
    })
  },
  
  startGameAnalysis: (pgn: string) => {
    set({
      isGameAnalyzing: true,
      gameAnalysis: null,
      gameAnalysisProgress: 0,
    })
  },
  
  setGameAnalysisResult: (result: GameAnalysis) => {
    set({
      gameAnalysis: result,
      isGameAnalyzing: false,
      gameAnalysisProgress: 100,
    })
  },
  
  setGameAnalysisProgress: (progress: number) => {
    set({
      gameAnalysisProgress: progress,
    })
  },
  
  // Engine actions
  setEngines: (engines: Record<string, EngineInfo>) => {
    set({ engines })
  },
  
  selectEngine: (engineName: string) => {
    set({ selectedEngine: engineName })
  },
  
  setEngineStatus: (engineName: string, status: 'online' | 'offline') => {
    set((state) => ({
      engineStatus: {
        ...state.engineStatus,
        [engineName]: status,
      },
    }))
  },
  
  // WebSocket actions
  setConnectionStatus: (status: 'connected' | 'connecting' | 'disconnected') => {
    set({
      connectionStatus: status,
      isConnected: status === 'connected',
    })
  },
  
  setLastMessage: (message: WebSocketMessage) => {
    set({ lastMessage: message })
  },
  
  incrementReconnectAttempts: () => {
    set((state) => ({
      reconnectAttempts: state.reconnectAttempts + 1,
    }))
  },
  
  resetReconnectAttempts: () => {
    set({ reconnectAttempts: 0 })
  },
  
  // UI actions
  toggleSidebar: () => {
    set((state) => ({ sidebarOpen: !state.sidebarOpen }))
  },
  
  setSidebarOpen: (open: boolean) => {
    set({ sidebarOpen: open })
  },
  
  setActiveTab: (tab: string) => {
    set({ activeTab: tab })
  },
  
  setTheme: (theme: 'light' | 'dark') => {
    set({ theme })
  },
  
  setBoardTheme: (theme: string) => {
    set({ boardTheme: theme })
  },
  
  setShowCoordinates: (show: boolean) => {
    set({ showCoordinates: show })
  },
  
  setShowMoveHints: (show: boolean) => {
    set({ showMoveHints: show })
  },
  
  setAutoAnalyze: (auto: boolean) => {
    set({ autoAnalyze: auto })
  },
  
  setAnalysisDepth: (depth: number) => {
    set({ analysisDepth: depth })
  },
}))

// Selectors for better performance
export const useChessState = () => useAppStore((state) => ({
  game: state.game,
  position: state.position,
  moveHistory: state.moveHistory,
  currentMoveIndex: state.currentMoveIndex,
  isFlipped: state.isFlipped,
}))

export const useAnalysisState = () => useAppStore((state) => ({
  currentAnalysis: state.currentAnalysis,
  isAnalyzing: state.isAnalyzing,
  analysisHistory: state.analysisHistory,
  gameAnalysis: state.gameAnalysis,
  isGameAnalyzing: state.isGameAnalyzing,
  gameAnalysisProgress: state.gameAnalysisProgress,
}))

export const useEngineState = () => useAppStore((state) => ({
  engines: state.engines,
  selectedEngine: state.selectedEngine,
  engineStatus: state.engineStatus,
}))

export const useWebSocketState = () => useAppStore((state) => ({
  isConnected: state.isConnected,
  connectionStatus: state.connectionStatus,
  lastMessage: state.lastMessage,
  reconnectAttempts: state.reconnectAttempts,
}))

export const useUIState = () => useAppStore((state) => ({
  sidebarOpen: state.sidebarOpen,
  activeTab: state.activeTab,
  theme: state.theme,
  boardTheme: state.boardTheme,
  showCoordinates: state.showCoordinates,
  showMoveHints: state.showMoveHints,
  autoAnalyze: state.autoAnalyze,
  analysisDepth: state.analysisDepth,
}))