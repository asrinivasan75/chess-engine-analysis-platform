import React from 'react';
import { Container, Typography, Card, CardContent } from '@mui/material';

const SettingsPage: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>
      
      <Card>
        <CardContent>
          <Typography variant="body1">
            Settings and preferences coming soon...
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            This page will include engine configuration, analysis preferences,
            theme selection, and other customization options.
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default SettingsPage;