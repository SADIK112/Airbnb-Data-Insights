import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';

const theme = createTheme({
  palette: {
    primary: {
      main: '#FF5A5F',
    },
    secondary: {
      main: '#008489',
    },
    background: {
      default: '#f7f7f7',
    },
  },
  typography: {
    fontFamily: 'Poppins, Arial, sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePrediction = async (formData) => {
    setLoading(true);
    setError('');
    
    try {
      // Prepare the data in the format expected by the API
      const requestData = {
        host_identity_verified: formData.host_identity_verified ? 1 : 0,
        instant_bookable: formData.instant_bookable ? 1 : 0,
        service_fee: parseFloat(formData.service_fee),
        minimum_nights: parseInt(formData.minimum_nights, 10),
        number_of_reviews: parseInt(formData.number_of_reviews, 10),
        reviews_per_month: parseFloat(formData.reviews_per_month),
        review_rate_number: parseFloat(formData.review_rate_number),
        calculated_host_listings_count: parseInt(formData.calculated_host_listings_count, 10),
        availability_365: parseInt(formData.availability_365, 10),
        'policy_flexible': formData.cancellation_policy === 'flexible' ? 1 : 0,
        'policy_moderate': formData.cancellation_policy === 'moderate' ? 1 : 0,
        'policy_strict': formData.cancellation_policy === 'strict' ? 1 : 0,
        'Entire home/apt': formData.property_type === 'entire_home' ? 1 : 0,
        'Hotel room': formData.property_type === 'hotel_room' ? 1 : 0,
        'Private room': formData.property_type === 'private_room' ? 1 : 0,
        'Shared room': formData.property_type === 'shared_room' ? 1 : 0,
        'neighbourhood_group_Bronx': formData.neighbourhood === 'Bronx' ? 1 : 0,
        'neighbourhood_group_Brooklyn': formData.neighbourhood === 'Brooklyn' ? 1 : 0,
        'neighbourhood_group_Manhattan': formData.neighbourhood === 'Manhattan' ? 1 : 0,
        'neighbourhood_group_Queens': formData.neighbourhood === 'Queens' ? 1 : 0,
        'neighbourhood_group_Staten Island': formData.neighbourhood === 'Staten Island' ? 1 : 0,
        'neighbourhood_group_Williamsburg': formData.neighbourhood === 'Williamsburg' ? 1 : 0,
        location_cluster: 3, // Default value, should be calculated based on location
        days_since_last_review: parseInt(formData.days_since_last_review, 10),
        availability_ratio: formData.availability_365 / 365,
        property_age: parseInt(formData.property_age, 10),
        has_house_rules: formData.has_house_rules ? 1 : 0,
        has_license: formData.has_license ? 1 : 0,
        popularity_score: parseFloat(formData.popularity_score),
        avg_reviews_per_listing: parseFloat(formData.avg_reviews_per_listing)
      };

      console.log('Sending data to API:', requestData);
      
      const response = await fetch('http://100.25.157.187:8080/price-predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({features: requestData}),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get prediction');
      }

      const data = await response.json();
      console.log('Received response from API:', data);
      setPrediction({
        predicted_price: data.prediction.prediction,
        details: formData
      });
    } catch (err) {
      setError(err.message || 'An error occurred while making the prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Airbnb Price Predictor
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Get an estimated price for your Airbnb listing
          </Typography>
        </Box>
        
        <PredictionForm onSubmit={handlePrediction} loading={loading} />
        
        {error && (
          <Box sx={{ mt: 3, p: 2, bgcolor: 'error.light', color: 'white', borderRadius: 1 }}>
            {error}
          </Box>
        )}
        
        {prediction && !loading && (
          <PredictionResult prediction={prediction} />
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;
