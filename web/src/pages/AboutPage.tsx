import React from 'react';
import { Container, Typography, Card, CardContent, Box, Chip } from '@mui/material';

const AboutPage: React.FC = () => {
  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        About Chess Engine Analysis Platform
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Overview
          </Typography>
          <Typography variant="body1" paragraph>
            This is a comprehensive chess analysis platform that combines modern web technologies
            with powerful chess engines to provide deep position analysis, game evaluation, 
            and training tools.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Features
          </Typography>
          <Box sx={{ mb: 2 }}>
            <Chip label="Real-time Position Analysis" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Multiple Chess Engines" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Game Analysis with PGN" sx={{ mr: 1, mb: 1 }} />
            <Chip label="WebSocket Real-time Updates" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Multi-PV Analysis" sx={{ mr: 1, mb: 1 }} />
            <Chip label="Move Classification" sx={{ mr: 1, mb: 1 }} />
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Technology Stack
          </Typography>
          <Typography variant="body2" component="div">
            <strong>Backend:</strong>
            <ul>
              <li>FastAPI (Python) - High-performance web framework</li>
              <li>python-chess - Chess library for position analysis</li>
              <li>Stockfish - World's strongest chess engine</li>
              <li>WebSocket support for real-time communication</li>
              <li>Async/await for concurrent processing</li>
            </ul>
            
            <strong>Frontend:</strong>
            <ul>
              <li>React 18 with TypeScript</li>
              <li>Material-UI for components and styling</li>
              <li>Zustand for state management</li>
              <li>Axios for API communication</li>
              <li>Framer Motion for animations</li>
            </ul>
            
            <strong>Chess Engine (In Development):</strong>
            <ul>
              <li>C++ with bitboard representation</li>
              <li>Alpha-beta search with optimizations</li>
              <li>Neural network evaluation (NNUE-style)</li>
              <li>UCI protocol compatibility</li>
            </ul>
          </Typography>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Development Status
          </Typography>
          <Typography variant="body1">
            This platform was built as a comprehensive chess analysis system. 
            The backend API and Stockfish integration are fully functional, 
            providing powerful position analysis capabilities.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Rating Background: Developed by someone with a chess.com rating of 2300 
            (USCF 99th percentile), ensuring the analysis features are designed 
            with practical chess knowledge in mind.
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default AboutPage;