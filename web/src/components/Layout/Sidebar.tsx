import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Typography,
  Box,
} from '@mui/material';
import {
  Home as HomeIcon,
  Analytics as AnalyticsIcon,
  SportsEsports as GameIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { Link, useLocation } from 'react-router-dom';

interface SidebarProps {
  open?: boolean;
  onClose?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ open = false, onClose }) => {
  const location = useLocation();
  
  const menuItems = [
    { path: '/', label: 'Home', icon: <HomeIcon /> },
    { path: '/analysis', label: 'Position Analysis', icon: <AnalyticsIcon /> },
    { path: '/game-analysis', label: 'Game Analysis', icon: <GameIcon /> },
    { path: '/about', label: 'About', icon: <InfoIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
  ];

  const drawerContent = (
    <Box sx={{ width: 280, mt: 2 }}>
      <Box sx={{ px: 2, mb: 2 }}>
        <Typography variant="h6" color="primary" fontWeight="bold">
          ♔ Chess Analysis
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Advanced position analysis platform
        </Typography>
      </Box>
      
      <Divider />
      
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              onClick={onClose}
              sx={{
                mx: 1,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
              }}
            >
              <ListItemIcon>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      
      <Divider sx={{ mt: 2 }} />
      
      <Box sx={{ p: 2, mt: 'auto' }}>
        <Typography variant="caption" color="text.secondary">
          Powered by Stockfish & FastAPI
        </Typography>
      </Box>
    </Box>
  );

  return (
    <>
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={open}
        onClose={onClose}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile
        }}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': { boxSizing: 'border-box' },
        }}
      >
        {drawerContent}
      </Drawer>
      
      {/* Desktop drawer - hidden for now since App.tsx doesn't expect it */}
    </>
  );
};

export default Sidebar;