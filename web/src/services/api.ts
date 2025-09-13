import axios from 'axios';

// API base URL - configure for development/production
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types (imported from store for consistency)
export interface AnalysisResult {
  fen: string;
  depth: number;
  score: string | number;
  pv: string[];
  nodes: number;
  time_ms: number;
  nps: number;
  multipv?: Array<{
    rank: number;
    score: string;
    score_cp: number;
    pv: string[];
    depth: number;
  }>;
  engine: string;
}

export interface EngineInfo {
  name: string;
  version: string;
  author: string;
  options: Record<string, any>;
  supported_features: string[];
}

// API functions
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get available engines
  async getEngines(): Promise<Record<string, EngineInfo>> {
    const response = await api.get('/engines');
    return response.data;
  },

  // Analyze position
  async analyzePosition(params: {
    fen: string;
    depth?: number;
    movetime?: number;
    nodes?: number;
    multipv?: number;
    engine?: string;
  }): Promise<AnalysisResult> {
    const response = await api.post('/analyze', {
      fen: params.fen,
      depth: params.depth || 15,
      movetime: params.movetime,
      nodes: params.nodes,
      multipv: params.multipv || 1,
      engine: params.engine || 'default'
    });
    return response.data;
  },

  // Analyze game
  async analyzeGame(params: {
    pgn: string;
    depth?: number;
    threshold?: number;
  }) {
    const response = await api.post('/analyze-game', {
      pgn: params.pgn,
      depth: params.depth || 15,
      threshold: params.threshold || 50
    });
    return response.data;
  },

  // Get game analysis results
  async getGameAnalysis(gameId: string) {
    const response = await api.get(`/analyze-game/${gameId}`);
    return response.data;
  }
};

// WebSocket connection for real-time analysis
export class AnalysisWebSocket {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 1000;
  private maxReconnectAttempts: number = 5;
  private reconnectAttempts: number = 0;

  constructor(
    private onMessage: (data: any) => void,
    private onConnect?: () => void,
    private onDisconnect?: () => void,
    private onError?: (error: any) => void
  ) {}

  connect() {
    try {
      const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws';
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.onConnect?.();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.onDisconnect?.();
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.onError?.(error);
      };
    } catch (error) {
      console.error('Error connecting WebSocket:', error);
      this.onError?.(error);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected');
    }
  }

  // Send analysis request
  analyzePosition(fen: string, depth: number = 15) {
    this.send({
      type: 'analyze',
      fen,
      depth
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      setTimeout(() => {
        console.log(`Attempting to reconnect WebSocket (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
        this.reconnectAttempts++;
        this.connect();
      }, this.reconnectInterval * Math.pow(2, this.reconnectAttempts));
    }
  }
}

export default apiService;