import React from 'react';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import {
  Box,
  Button,
  TextField,
  MenuItem,
  Grid,
  Paper,
  Typography,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  FormHelperText,
  FormControlLabel,
  Checkbox,
} from '@mui/material';

const propertyTypes = [
  { value: 'entire_home', label: 'Entire home/apt' },
  { value: 'private_room', label: 'Private room' },
  { value: 'shared_room', label: 'Shared room' },
  { value: 'hotel_room', label: 'Hotel room' },
];

const neighbourhoods = [
  { value: 'Manhattan', label: 'Manhattan' },
  { value: 'Brooklyn', label: 'Brooklyn' },
  { value: 'Queens', label: 'Queens' },
  { value: 'Bronx', label: 'Bronx' },
  { value: 'Staten Island', label: 'Staten Island' },
  { value: 'Williamsburg', label: 'Williamsburg' },
];

const cancellationPolicies = [
  { value: 'flexible', label: 'Flexible' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'strict', label: 'Strict' },
];

const validationSchema = Yup.object({
  // Property Details
  property_type: Yup.string().required('Required'),
  host_identity_verified: Yup.boolean().required('Required'),
  instant_bookable: Yup.boolean().required('Required'),
  cancellation_policy: Yup.string().required('Required'),
  
  // Location
  neighbourhood: Yup.string().required('Required'),
  
  // Booking Details
  minimum_nights: Yup.number()
    .required('Required')
    .min(1, 'Minimum 1 night')
    .max(365, 'Maximum 365 nights'),
  
  // Reviews
  number_of_reviews: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
  review_rate_number: Yup.number()
    .required('Required')
    .min(0, 'Minimum 0')
    .max(5, 'Maximum 5'),
  reviews_per_month: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
  days_since_last_review: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
  
  // Host Information
  calculated_host_listings_count: Yup.number()
    .required('Required')
    .min(1, 'Minimum 1'),
  
  // Property Features
  service_fee: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
  availability_365: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative')
    .max(365, 'Maximum 365 days'),
  
  // Additional Features
  property_age: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
  has_house_rules: Yup.boolean().required('Required'),
  has_license: Yup.boolean().required('Required'),
  popularity_score: Yup.number()
    .required('Required')
    .min(1, 'Minimum 1')
    .max(10, 'Maximum 10'),
  avg_reviews_per_listing: Yup.number()
    .required('Required')
    .min(0, 'Cannot be negative'),
});

const PredictionForm = ({ onSubmit, loading }) => {
  const formik = useFormik({
    initialValues: {
      // Property Details
      property_type: 'entire_home',
      host_identity_verified: true,
      instant_bookable: true,
      cancellation_policy: 'moderate',
      
      // Location
      neighbourhood: 'Manhattan',
      
      // Booking Details
      minimum_nights: 3,
      
      // Reviews
      number_of_reviews: 24,
      review_rate_number: 4.8,
      reviews_per_month: 2.5,
      days_since_last_review: 45,
      
      // Host Information
      calculated_host_listings_count: 2,
      
      // Property Features
      service_fee: 75.50,
      availability_365: 200,
      
      // Additional Features
      property_age: 5,
      has_house_rules: true,
      has_license: true,
      popularity_score: 7.5,
      avg_reviews_per_listing: 3.2,
    },
    validationSchema,
    onSubmit: (values) => {
      onSubmit(values);
    },
  });

  return (
    <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
      <Typography variant="h6" gutterBottom>
        Property Details
      </Typography>
      <form onSubmit={formik.handleSubmit}>
        <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
          Property Information
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth error={formik.touched.property_type && Boolean(formik.errors.property_type)}>
              <InputLabel>Property Type</InputLabel>
              <Select
                name="property_type"
                value={formik.values.property_type}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                label="Property Type"
              >
                {propertyTypes.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {formik.touched.property_type && formik.errors.property_type}
              </FormHelperText>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth error={formik.touched.neighbourhood && Boolean(formik.errors.neighbourhood)}>
              <InputLabel>Neighborhood</InputLabel>
              <Select
                name="neighbourhood"
                value={formik.values.neighbourhood}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                label="Neighborhood"
              >
                {neighbourhoods.map((neighborhood) => (
                  <MenuItem key={neighborhood.value} value={neighborhood.value}>
                    {neighborhood.label}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {formik.touched.neighbourhood && formik.errors.neighbourhood}
              </FormHelperText>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <FormControl fullWidth error={formik.touched.cancellation_policy && Boolean(formik.errors.cancellation_policy)}>
              <InputLabel>Cancellation Policy</InputLabel>
              <Select
                name="cancellation_policy"
                value={formik.values.cancellation_policy}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                label="Cancellation Policy"
              >
                {cancellationPolicies.map((policy) => (
                  <MenuItem key={policy.value} value={policy.value}>
                    {policy.label}
                  </MenuItem>
                ))}
              </Select>
              <FormHelperText>
                {formik.touched.cancellation_policy && formik.errors.cancellation_policy}
              </FormHelperText>
            </FormControl>
          </Grid>

          {/* Host Verification & Booking */}
          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Checkbox
                  name="host_identity_verified"
                  checked={formik.values.host_identity_verified}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  color="primary"
                />
              }
              label="Host Identity Verified"
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Checkbox
                  name="instant_bookable"
                  checked={formik.values.instant_bookable}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  color="primary"
                />
              }
              label="Instant Bookable"
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Checkbox
                  name="has_house_rules"
                  checked={formik.values.has_house_rules}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  color="primary"
                />
              }
              label="Has House Rules"
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <FormControlLabel
              control={
                <Checkbox
                  name="has_license"
                  checked={formik.values.has_license}
                  onChange={formik.handleChange}
                  onBlur={formik.handleBlur}
                  color="primary"
                />
              }
              label="Has Valid License"
            />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Property Details
            </Typography>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="service_fee"
              name="service_fee"
              label="Service Fee ($)"
              type="number"
              value={formik.values.service_fee}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.service_fee && Boolean(formik.errors.service_fee)}
              helperText={formik.touched.service_fee && formik.errors.service_fee}
              inputProps={{ step: '0.01' }}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="minimum_nights"
              name="minimum_nights"
              label="Minimum Nights"
              type="number"
              value={formik.values.minimum_nights}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.minimum_nights && Boolean(formik.errors.minimum_nights)}
              helperText={formik.touched.minimum_nights && formik.errors.minimum_nights}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="availability_365"
              name="availability_365"
              label="Days Available (Next 365 Days)"
              type="number"
              value={formik.values.availability_365}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.availability_365 && Boolean(formik.errors.availability_365)}
              helperText={formik.touched.availability_365 && formik.errors.availability_365}
            />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Reviews & Ratings
            </Typography>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              id="number_of_reviews"
              name="number_of_reviews"
              label="Total Reviews"
              type="number"
              value={formik.values.number_of_reviews}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.number_of_reviews && Boolean(formik.errors.number_of_reviews)}
              helperText={formik.touched.number_of_reviews && formik.errors.number_of_reviews}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              id="review_rate_number"
              name="review_rate_number"
              label="Rating (0-5)"
              type="number"
              value={formik.values.review_rate_number}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.review_rate_number && Boolean(formik.errors.review_rate_number)}
              helperText={formik.touched.review_rate_number && formik.errors.review_rate_number}
              inputProps={{ step: '0.1', min: '0', max: '5' }}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              id="reviews_per_month"
              name="reviews_per_month"
              label="Reviews Per Month"
              type="number"
              value={formik.values.reviews_per_month}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.reviews_per_month && Boolean(formik.errors.reviews_per_month)}
              helperText={formik.touched.reviews_per_month && formik.errors.reviews_per_month}
              inputProps={{ step: '0.1' }}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <TextField
              fullWidth
              id="days_since_last_review"
              name="days_since_last_review"
              label="Days Since Last Review"
              type="number"
              value={formik.values.days_since_last_review}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.days_since_last_review && Boolean(formik.errors.days_since_last_review)}
              helperText={formik.touched.days_since_last_review && formik.errors.days_since_last_review}
            />
          </Grid>

          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
              Host & Property Metrics
            </Typography>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="calculated_host_listings_count"
              name="calculated_host_listings_count"
              label="Host's Total Listings"
              type="number"
              value={formik.values.calculated_host_listings_count}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.calculated_host_listings_count && Boolean(formik.errors.calculated_host_listings_count)}
              helperText={formik.touched.calculated_host_listings_count && formik.errors.calculated_host_listings_count}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="property_age"
              name="property_age"
              label="Property Age (Years)"
              type="number"
              value={formik.values.property_age}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.property_age && Boolean(formik.errors.property_age)}
              helperText={formik.touched.property_age && formik.errors.property_age}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="popularity_score"
              name="popularity_score"
              label="Popularity Score (1-10)"
              type="number"
              value={formik.values.popularity_score}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.popularity_score && Boolean(formik.errors.popularity_score)}
              helperText={formik.touched.popularity_score && formik.errors.popularity_score}
              inputProps={{ min: '1', max: '10', step: '0.1' }}
            />
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <TextField
              fullWidth
              id="avg_reviews_per_listing"
              name="avg_reviews_per_listing"
              label="Avg. Reviews per Listing"
              type="number"
              value={formik.values.avg_reviews_per_listing}
              onChange={formik.handleChange}
              onBlur={formik.handleBlur}
              error={formik.touched.avg_reviews_per_listing && Boolean(formik.errors.avg_reviews_per_listing)}
              helperText={formik.touched.avg_reviews_per_listing && formik.errors.avg_reviews_per_listing}
              inputProps={{ step: '0.1' }}
            />
          </Grid>


          <Grid item xs={12}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={loading}
              fullWidth
              size="large"
              sx={{ mt: 2 }}
            >
              {loading ? (
                <>
                  <CircularProgress size={24} color="inherit" sx={{ mr: 1 }} />
                  Predicting...
                </>
              ) : (
                'Predict Price'
              )}
            </Button>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default PredictionForm;
