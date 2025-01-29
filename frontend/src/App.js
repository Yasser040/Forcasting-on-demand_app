import React, { useState } from 'react';
import axios from 'axios';
import FormInputs from './components/FormInputs';
import ForecastResults from './components/ForecastResults';
import './index.css';
import logo from './assets/logo.jpg'; // Importing the logo

const App = () => {
  const [formData, setFormData] = useState({
    base_price: '',
    total_price: '',
    is_featured_sku: false,
    is_display_sku: false,
    sku_id: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        base_price: parseFloat(formData.base_price),
        total_price: parseFloat(formData.total_price),
        is_featured_sku: formData.is_featured_sku ? 1 : 0,
        is_display_sku: formData.is_display_sku ? 1 : 0,
        sku_id: parseInt(formData.sku_id, 10)
      });
      setPrediction(response.data.predicted_units);
      setAnalysis(response.data.metadata.analysis);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setAnalysis('An error occurred while fetching the prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <div className="container">
        <img src={logo} alt="Logo" className="logo" />
        <h1 style={{ marginBottom: '20px' }}>Demand Forecasting Application</h1>
        <form onSubmit={handleSubmit}>
          <FormInputs formData={formData} handleChange={handleChange} />
          <button
            type="submit"
            disabled={loading}
            className="button"
          >
            {loading ? 'Processing...' : 'Generate Forecast'}
          </button>
        </form>
        <ForecastResults
          loading={loading}
          prediction={prediction}
          analysis={analysis}
        />
      </div>
      <footer className="footer">
        © 2025 Team Divergent
      </footer>
    </div>
  );
};

export default App;