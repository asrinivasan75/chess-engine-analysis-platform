import React, { useState } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { Container, Box } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'

// Components
import Header from './components/Layout/Header'
import Sidebar from './components/Layout/Sidebar'
import WebSocketStatus from './components/Common/WebSocketStatus'

// Pages
import HomePage from './pages/HomePage'
import AnalysisPage from './pages/AnalysisPage'
import GameAnalysisPage from './pages/GameAnalysisPage'
import AboutPage from './pages/AboutPage'
import SettingsPage from './pages/SettingsPage'

// Store
import { useAppStore } from './store/appStore'

// Types
interface AppProps {}

const App: React.FC<AppProps> = () => {
  const location = useLocation()
  const { sidebarOpen } = useAppStore()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
        backgroundColor: 'background.default',
      }}
    >
      {/* Header */}
      <Header onMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)} />
      
      {/* WebSocket Status */}
      <WebSocketStatus />
      
      {/* Sidebar */}
      <Sidebar 
        open={mobileMenuOpen} 
        onClose={() => setMobileMenuOpen(false)} 
      />
      
      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          position: 'relative',
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{
              duration: 0.3,
              ease: 'easeInOut',
            }}
            style={{ height: '100%' }}
          >
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/game-analysis" element={<GameAnalysisPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </Box>
    </Box>
  )
}

export default App