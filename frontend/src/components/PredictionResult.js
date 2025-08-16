import React from 'react';
import { Box, Typography, Paper, Grid, Divider, useTheme } from '@mui/material';
import LocalHotelIcon from '@mui/icons-material/LocalHotel';
import BathtubIcon from '@mui/icons-material/Bathtub';
import MeetingRoomIcon from '@mui/icons-material/MeetingRoom';
import PeopleIcon from '@mui/icons-material/People';
import NightsStayIcon from '@mui/icons-material/NightsStay';
import LocationOnIcon from '@mui/icons-material/LocationOn';

const PredictionResult = ({ prediction }) => {
  const theme = useTheme();
  
  if (!prediction) return null;

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence > 0.8) return 'High';
    if (confidence > 0.6) return 'Moderate';
    return 'Low';
  };

  return (
    <Paper elevation={3} sx={{ p: 4, mt: 3 }}>
      <Typography variant="h5" component="div" gutterBottom sx={{ textAlign: 'center' }}>
        🏠 Predicted Price
      </Typography>
      <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold', my: 2, textAlign: 'center' }}>
        ${prediction.predicted_price.toFixed(2) || 'N/A'}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
        per night
      </Typography>
      
      <Box sx={{ 
        display: 'inline-block',
        bgcolor: theme.palette.primary.light,
        color: 'white',
        px: 2,
        py: 1,
        borderRadius: 1,
        mb: 3
      }}>
        <Typography variant="body2">
          Confidence: {getConfidenceLevel(prediction.confidence || 0.8)} • ±{formatCurrency((prediction.confidence || 0.8) * 20)}
        </Typography>
      </Box>
      
      <Grid container spacing={3} sx={{ mb: 2 }}>
        <Grid item xs={6} sm={3}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <MeetingRoomIcon color="primary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="body2" color="textSecondary">Bedrooms</Typography>
              <Typography variant="body1">{prediction.details?.bedrooms || 1}</Typography>
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={6} sm={3}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <LocalHotelIcon color="primary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="body2" color="textSecondary">Beds</Typography>
              <Typography variant="body1">{prediction.details?.beds || 1}</Typography>
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={6} sm={3}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <BathtubIcon color="primary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="body2" color="textSecondary">Bathrooms</Typography>
              <Typography variant="body1">{prediction.details?.bathrooms || 1}</Typography>
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={6} sm={3}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <PeopleIcon color="primary" sx={{ mr: 1 }} />
            <Box>
              <Typography variant="body2" color="textSecondary">Guests</Typography>
              <Typography variant="body1">{prediction.details?.accommodates || 2}</Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
      
      <Divider sx={{ my: 3 }} />
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', mb: 2 }}>
            <LocationOnIcon color="primary" sx={{ mr: 1, mt: 0.5 }} />
            <Box>
              <Typography variant="subtitle2" color="textSecondary">Location</Typography>
              <Typography variant="body1">
                {prediction.details?.neighbourhood || 'Not specified'}
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', mb: 2 }}>
            <NightsStayIcon color="primary" sx={{ mr: 1, mt: 0.5 }} />
            <Box>
              <Typography variant="subtitle2" color="textSecondary">Minimum Stay</Typography>
              <Typography variant="body1">
                {prediction.details?.minimum_nights || 1} night{prediction.details?.minimum_nights !== 1 ? 's' : ''}
              </Typography>
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" color="textSecondary" gutterBottom>
            Included Amenities
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {(prediction.details?.amenities || []).map((amenity, index) => (
              <Box 
                key={index}
                sx={{
                  bgcolor: theme.palette.grey[100],
                  px: 1.5,
                  py: 0.5,
                  borderRadius: 1,
                  display: 'inline-flex',
                  alignItems: 'center'
                }}
              >
                <Typography variant="body2">
                  {amenity.charAt(0).toUpperCase() + amenity.slice(1).replace('_', ' ')}
                </Typography>
              </Box>
            ))}
          </Box>
        </Grid>
      </Grid>
      
      <Box sx={{ mt: 3, p: 2, bgcolor: theme.palette.info.light, borderRadius: 1 }}>
        <Typography variant="body2" color="textSecondary">
          <strong>Note:</strong> This is an estimate based on similar listings. Actual prices may vary.
        </Typography>
      </Box>
    </Paper>
  );
};

export default PredictionResult;
