import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { apiService, AnalysisResult, EngineInfo } from '../services/api';

const HomePage: React.FC = () => {
  const [engines, setEngines] = useState<Record<string, EngineInfo>>({});
  const [position, setPosition] = useState('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1');
  const [depth, setDepth] = useState(15);
  const [multipv, setMultipv] = useState(3);
  const [selectedEngine, setSelectedEngine] = useState('stockfish');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load engines on component mount
  useEffect(() => {
    loadEngines();
  }, []);

  const loadEngines = async () => {
    try {
      setIsLoading(true);
      const enginesData = await apiService.getEngines();
      setEngines(enginesData);
      
      // Set default engine if available
      if (enginesData.stockfish) {
        setSelectedEngine('stockfish');
      } else if (enginesData.default) {
        setSelectedEngine('default');
      }
      
      setError(null);
    } catch (err: any) {
      setError(`Failed to load engines: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzePosition = async () => {
    if (!position.trim()) {
      setError('Please enter a valid FEN position');
      return;
    }

    try {
      setIsAnalyzing(true);
      setError(null);
      
      const result = await apiService.analyzePosition({
        fen: position,
        depth: depth,
        multipv: multipv,
        engine: selectedEngine
      });
      
      setAnalysis(result);
    } catch (err: any) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatScore = (score: string | number) => {
    if (typeof score === 'string' && score.includes('mate')) {
      return score;
    }
    const numScore = Number(score);
    if (numScore > 0) {
      return `+${(numScore / 100).toFixed(2)}`;
    } else {
      return (numScore / 100).toFixed(2);
    }
  };

  const getScoreColor = (score: string | number) => {
    if (typeof score === 'string' && score.includes('mate')) {
      return score.includes('-') ? '#f44336' : '#4caf50';
    }
    const numScore = Number(score);
    if (numScore > 100) return '#4caf50';
    if (numScore < -100) return '#f44336';
    return '#ff9800';
  };

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading chess engines...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center" color="primary">
        Chess Engine Analysis Platform
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Engine Information */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Available Engines
              </Typography>
              {Object.keys(engines).length === 0 ? (
                <Alert severity="warning">
                  No engines available. Make sure the backend is running and engines are configured.
                </Alert>
              ) : (
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  {Object.entries(engines).map(([key, engine]) => (
                    <Chip
                      key={key}
                      label={`${engine.name} (${key})`}
                      color={key === selectedEngine ? 'primary' : 'default'}
                      onClick={() => setSelectedEngine(key)}
                      variant={key === selectedEngine ? 'filled' : 'outlined'}
                    />
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Input */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Position Analysis
              </Typography>
              
              <TextField
                fullWidth
                label="FEN Position"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                margin="normal"
                multiline
                rows={2}
                helperText="Enter FEN string of the position to analyze"
              />
              
              <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                <TextField
                  label="Depth"
                  type="number"
                  value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))}
                  inputProps={{ min: 1, max: 30 }}
                  sx={{ width: 100 }}
                />
                <TextField
                  label="Multi PV"
                  type="number"
                  value={multipv}
                  onChange={(e) => setMultipv(Number(e.target.value))}
                  inputProps={{ min: 1, max: 5 }}
                  sx={{ width: 100 }}
                />
              </Box>
              
              <Button
                variant="contained"
                onClick={analyzePosition}
                disabled={isAnalyzing || Object.keys(engines).length === 0}
                sx={{ mt: 3 }}
                fullWidth
                size="large"
              >
                {isAnalyzing ? (
                  <>
                    <CircularProgress size={20} sx={{ mr: 1 }} />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Position'
                )}
              </Button>
              
              {/* Preset Positions */}
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Quick Test Positions:
                </Typography>
                <Button
                  size="small"
                  onClick={() => setPosition('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')}
                  sx={{ mr: 1, mb: 1 }}
                >
                  Starting Position (1.e4)
                </Button>
                <Button
                  size="small"
                  onClick={() => setPosition('r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4')}
                  sx={{ mr: 1, mb: 1 }}
                >
                  Italian Game
                </Button>
                <Button
                  size="small"
                  onClick={() => setPosition('8/8/8/8/3k4/8/3K4/8 w - - 0 1')}
                  sx={{ mr: 1, mb: 1 }}
                >
                  King Endgame
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Analysis Results */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              
              {!analysis && (
                <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  No analysis yet. Enter a position and click "Analyze Position" to get started.
                </Typography>
              )}
              
              {analysis && (
                <Box>
                  {/* Main Analysis */}
                  <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: 1, borderColor: 'divider' }}>
                    <Typography variant="subtitle1" gutterBottom>
                      <strong>Best Move:</strong> {analysis.pv[0] || 'None'}
                    </Typography>
                    <Typography variant="subtitle1" gutterBottom>
                      <strong>Evaluation:</strong> 
                      <Chip
                        label={formatScore(analysis.score)}
                        size="small"
                        sx={{ 
                          ml: 1, 
                          backgroundColor: getScoreColor(analysis.score),
                          color: 'white',
                          fontWeight: 'bold'
                        }}
                      />
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Engine:</strong> {analysis.engine} | 
                      <strong> Depth:</strong> {analysis.depth} | 
                      <strong> Nodes:</strong> {analysis.nodes.toLocaleString()} | 
                      <strong> Speed:</strong> {(analysis.nps / 1000).toFixed(0)}k nps |
                      <strong> Time:</strong> {analysis.time_ms}ms
                    </Typography>
                  </Box>

                  {/* Principal Variation */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Principal Variation:
                    </Typography>
                    <Typography variant="body2" sx={{ 
                      fontFamily: 'monospace', 
                      backgroundColor: 'grey.100', 
                      p: 1, 
                      borderRadius: 1 
                    }}>
                      {analysis.pv.join(' ')}
                    </Typography>
                  </Box>

                  {/* Multi PV Results */}
                  {analysis.multipv && analysis.multipv.length > 1 && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="subtitle2">
                          Multiple Variations ({analysis.multipv.length} lines)
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Rank</TableCell>
                                <TableCell>Score</TableCell>
                                <TableCell>Principal Variation</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {analysis.multipv.map((line) => (
                                <TableRow key={line.rank}>
                                  <TableCell>{line.rank}</TableCell>
                                  <TableCell>
                                    <Chip
                                      label={formatScore(line.score)}
                                      size="small"
                                      sx={{ 
                                        backgroundColor: getScoreColor(line.score_cp),
                                        color: 'white',
                                        fontWeight: 'bold'
                                      }}
                                    />
                                  </TableCell>
                                  <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                                    {line.pv.join(' ')}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </AccordionDetails>
                    </Accordion>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default HomePage;