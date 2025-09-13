import React from 'react';
import { Container, Typography, Card, CardContent } from '@mui/material';

const GameAnalysisPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Game Analysis
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="body1">
            Game analysis features coming soon...
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            This page will allow you to upload PGN files and analyze complete games,
            showing move classifications, accuracy scores, and critical moments.
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default GameAnalysisPage;