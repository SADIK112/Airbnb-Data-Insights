# Airbnb Price Predictor Frontend

This is a React-based frontend for the Airbnb Price Predictor application. It provides a user-friendly interface for users to input property details and get price predictions.

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher) or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

### Running the Development Server

```bash
npm start
# or
yarn start
```

This will start the development server at [http://localhost:3000](http://localhost:3000).

### Connecting to the Backend

By default, the frontend is configured to connect to a backend server at `http://localhost:8000`. To change this, update the API URL in `src/App.js`.

## Features

- **Interactive Form**: User-friendly form to input property details
- **Real-time Validation**: Form validation with helpful error messages
- **Responsive Design**: Works on desktop and mobile devices
- **Detailed Results**: Clear presentation of prediction results with confidence levels
- **Modern UI**: Built with Material-UI for a polished look and feel

## Project Structure

```
frontend/
├── public/
│   └── index.html          # Main HTML template
├── src/
│   ├── components/         # Reusable components
│   │   ├── PredictionForm.js  # Form for user input
│   │   └── PredictionResult.js # Component to display results
│   ├── App.js              # Main application component
│   └── index.js            # Application entry point
├── package.json            # Project dependencies and scripts
└── README.md               # This file
```

## Available Scripts

- `npm start` or `yarn start`: Runs the app in development mode
- `npm test` or `yarn test`: Launches the test runner
- `npm run build` or `yarn build`: Builds the app for production
- `npm run eject` or `yarn eject`: Ejects from Create React App (advanced)

## Dependencies

- React 18
- Material-UI 5
- Formik & Yup for form handling and validation
- Axios for HTTP requests
- Recharts for data visualization

## License

This project is licensed under the MIT License.
