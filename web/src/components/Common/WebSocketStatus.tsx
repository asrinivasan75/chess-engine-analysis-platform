import React, { useState, useEffect } from 'react';
import { Chip, Alert } from '@mui/material';
import { AnalysisWebSocket } from '../../services/api';

const WebSocketStatus: React.FC = () => {
  const [status, setStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [ws, setWs] = useState<AnalysisWebSocket | null>(null);

  useEffect(() => {
    const websocket = new AnalysisWebSocket(
      (data) => {
        console.log('WebSocket message:', data);
        // Handle WebSocket messages here
      },
      () => {
        setStatus('connected');
      },
      () => {
        setStatus('disconnected');
      },
      (error) => {
        console.error('WebSocket error:', error);
        setStatus('disconnected');
      }
    );

    setWs(websocket);
    setStatus('connecting');
    websocket.connect();

    return () => {
      websocket.disconnect();
    };
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'disconnected': return 'error';
      default: return 'default';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected': return 'Real-time Connected';
      case 'connecting': return 'Connecting...';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  return (
    <Chip
      label={getStatusText()}
      color={getStatusColor()}
      size="small"
      sx={{
        position: 'fixed',
        top: 80,
        right: 20,
        zIndex: 1000,
      }}
    />
  );
};

export default WebSocketStatus;