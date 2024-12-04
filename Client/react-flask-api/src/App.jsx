import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import PredictionMaker from './PredictionMaker';
import PredictionsTable from './PredictionsTable'; // Ensure this component is created

function App() {
  return (
    <Router>
      <div className="container mt-4">
        <nav>
          <ul className="nav nav-pills">
            <li className="nav-item">
              <Link className="nav-link" to="/">Make a Prediction</Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/predictions">View Prediction History</Link>
            </li>
          </ul>
        </nav>

        <Routes>
          <Route path="/" element={<PredictionMaker />} />
          <Route path="/predictions" element={<PredictionsTable />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;