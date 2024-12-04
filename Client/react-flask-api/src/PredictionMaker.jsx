import React, { useState } from 'react';
import axios from 'axios';
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [formData, setFormData] = useState({
    MANOEUVER_Going_Ahead: 0,
    DRIVACT_Driving_Properly: 0,
    IMPACTYPE_Pedestrian_Collisions: 0,
    INVTYPE_Driver: 0,
    DRIVCOND_Unknown: 0,
    VEHTYPE_Automobile_Station_Wagon: 0,
    HOUR: "",
    YEAR: "",
    INVTYPE_Pedestrian: 0,
    INVTYPE_Passenger: 0,
    VEHTYPE_Other: 0,
    INVAGE_unknown: 0,
    INITDIR_West: 0,
    DRIVCOND_Normal: 0,
    INITDIR_South: 0,
    ROAD_CLASS_Major_Arterial: 0,
    INITDIR_North: 0
  });

  // Handle input changes for form fields
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (name.startsWith("INITDIR")) {
      // Handle mutually exclusive options for INITDIR
      setFormData({
        ...formData,
        INITDIR_North: 0,
        INITDIR_South: 0,
        INITDIR_West: 0,
        [name]: 1,
      });
    } else if (name === "DRIVCOND") {
      // Handle mutually exclusive options for Driver Condition
      setFormData({
        ...formData,
        DRIVCOND_Normal: value === "Normal" ? 1 : 0,
        DRIVCOND_Unknown: value === "Unknown" ? 1 : 0,
      });
    } else if (name === "INVAGE_unknown") {
      // Handle Age Known Checkbox
      setFormData({
        ...formData,
        INVAGE_unknown: checked ? 0 : 1, // 0 if checked (age known), 1 if unchecked (unknown)
      });
    } else if (name === "IMPACTYPE_Pedestrian_Collisions") {
      // Handle mutually exclusive options for IMPACTYPE
      setFormData({
        ...formData,
        IMPACTYPE_Pedestrian_Collisions: value === "Yes" ? 1 : 0,
      });
    } else if (name === "INVTYPE") {
      // Handle mutually exclusive options for INVTYPE
      setFormData({
        ...formData,
        INVTYPE_Pedestrian: value === "Pedestrian" ? 1 : 0,
        INVTYPE_Passenger: value === "Passenger" ? 1 : 0,
        INVTYPE_Driver: value === "Driver" ? 1 : 0,
      });
    } else if (name === "DRIVCOND_Normal" || name === "DRIVCOND_Unknown") {
      // Handle mutually exclusive options for DRIVCOND
      setFormData({
        ...formData,
        DRIVCOND_Normal: name === "DRIVCOND_Normal" ? 1 : 0,
        DRIVCOND_Unknown: name === "DRIVCOND_Unknown" ? 1 : 0,
      });
    } else if (name === "VEHTYPE_Automobile_Station_Wagon" || name === "VEHTYPE_Other") {
      // Handle mutually exclusive options for VEHTYPE
      setFormData({
        ...formData,
        VEHTYPE_Automobile_Station_Wagon: name === "VEHTYPE_Automobile_Station_Wagon" ? 1 : 0,
        VEHTYPE_Other: name === "VEHTYPE_Other" ? 1 : 0,
      });
    } else {
      // Handle general inputs and checkboxes
      setFormData({
        ...formData,
        [name]: type === "checkbox" ? (checked ? 1 : 0) : value,
      });
    }
  };

  // Submit form data
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate required fields
    if (!formData.HOUR || !formData.YEAR) {
      alert("Please fill out all required fields.");
      return;
    }

    // Prepare payload for the API
    const payload = {
      ...formData,
      HOUR: parseInt(formData.HOUR, 10),
      YEAR: parseInt(formData.YEAR, 10),
    };

    console.log("Payload being sent:", payload);

    try {
      // Send data to the Flask API
      const response = await axios.post("http://127.0.0.1:5000/predict", payload);
      alert(`Prediction: ${response.data.prediction}, Probability: ${response.data.probability.toFixed(2)}`);
    } catch (error) {
      console.error("Error submitting form:", error.response?.data || error);
      alert("Failed to get prediction. Check console for details.");
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4">Fatal Accident Predictor</h1>
      <form onSubmit={handleSubmit}>
        {/* HOUR Input */}
        <div className="mb-3">
          <label>Hour:</label>
          <input
            type="number"
            className="form-control"
            name="HOUR"
            value={formData.HOUR}
            onChange={handleChange}
            min="0"
            max="23"
            required
          />
        </div>
  
        {/* YEAR Input */}
        <div className="mb-3">
          <label>Year:</label>
          <input
            type="number"
            className="form-control"
            name="YEAR"
            value={formData.YEAR}
            onChange={handleChange}
            min="2000"
            max="2024"
            required
          />
        </div>
  
        {/* MANOEUVER_Going_Ahead */}
        <div className="mb-3">
          <label>Was the Vehicle Maneuver Going Ahead?</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="MANOEUVER_Going_Ahead"
              checked={formData.MANOEUVER_Going_Ahead === 1}
              onChange={() => setFormData({ ...formData, MANOEUVER_Going_Ahead: 1 })}
            />
            <label className="form-check-label">Yes</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="MANOEUVER_Going_Ahead"
              checked={formData.MANOEUVER_Going_Ahead === 0}
              onChange={() => setFormData({ ...formData, MANOEUVER_Going_Ahead: 0 })}
            />
            <label className="form-check-label">No</label>
          </div>
        </div>
  
        {/* DRIVACT_Driving_Properly */}
        <div className="mb-3">
          <label>Apparent Driver Action:</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="DRIVACT_Driving_Properly"
              checked={formData.DRIVACT_Driving_Properly === 1}
              onChange={() => setFormData({ ...formData, DRIVACT_Driving_Properly: 1 })}
            />
            <label className="form-check-label">Proper</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="DRIVACT_Driving_Properly"
              checked={formData.DRIVACT_Driving_Properly === 0}
              onChange={() => setFormData({ ...formData, DRIVACT_Driving_Properly: 0 })}
            />
            <label className="form-check-label">Improper</label>
          </div>
        </div>

        {/* Driver Condition Radio Buttons */}
        <div className="mb-3">
          <label>Driver Condition:</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="DRIVCOND"
              value="Normal"
              checked={formData.DRIVCOND_Normal === 1}
              onChange={() =>
                setFormData({
                  ...formData,
                  DRIVCOND_Normal: 1,
                  DRIVCOND_Unknown: 0,
                })
              }
            />
            <label className="form-check-label">Normal</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="DRIVCOND"
              value="Unknown"
              checked={formData.DRIVCOND_Unknown === 1}
              onChange={() =>
                setFormData({
                  ...formData,
                  DRIVCOND_Normal: 0,
                  DRIVCOND_Unknown: 1,
                })
              }
            />
            <label className="form-check-label">Unknown</label>
          </div>
        </div>

        {/* Age Known Checkbox */}
        <div className="mb-3">
          <label>Is the Age of the Involved Person Known?</label>
          <div className="form-check">
            <input
              type="checkbox"
              className="form-check-input"
              name="INVAGE_unknown"
              checked={formData.INVAGE_unknown === 0}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  INVAGE_unknown: e.target.checked ? 0 : 1,
                })
              }
            />
            <label className="form-check-label">Yes</label>
          </div>
        </div>
  
        {/* IMPACTYPE_Pedestrian_Collisions */}
        <div className="mb-3">
          <label>Was the Initial Impact with a Pedestrian?</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="IMPACTYPE_Pedestrian_Collisions"
              checked={formData.IMPACTYPE_Pedestrian_Collisions === 1}
              onChange={() => setFormData({ ...formData, IMPACTYPE_Pedestrian_Collisions: 1 })}
            />
            <label className="form-check-label">Yes</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="IMPACTYPE_Pedestrian_Collisions"
              checked={formData.IMPACTYPE_Pedestrian_Collisions === 0}
              onChange={() => setFormData({ ...formData, IMPACTYPE_Pedestrian_Collisions: 0 })}
            />
            <label className="form-check-label">No</label>
          </div>
        </div>

        {/* INVTYPE - Involvement Type Radio Box To Choose Driver, Passenger or Pedestrian*/}
        <div className="mb-3">
          <label>Involvement Type:</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="INVTYPE"
              value="Driver"
              checked={formData.INVTYPE_Driver === 1}
              onChange={(e) => handleChange(e)}
            />
            <label className="form-check-label">Driver</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="INVTYPE"
              value="Passenger"
              checked={formData.INVTYPE_Passenger === 1}
              onChange={(e) => handleChange(e)}
            />
            <label className="form-check-label">Passenger</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="INVTYPE"
              value="Pedestrian"
              checked={formData.INVTYPE_Pedestrian === 1}
              onChange={(e) => handleChange(e)}
            />
            <label className="form-check-label">Pedestrian</label>
          </div>
        </div>
  
        {/* INITDIR */}
        <div className="mb-3">
          <label>Initial Direction of Travel:</label>
          <select
            className="form-select"
            name="INITDIR"
            value={
              formData.INITDIR_North
                ? "North"
                : formData.INITDIR_South
                ? "South"
                : formData.INITDIR_West
                ? "West"
                : ""
            }
            onChange={(e) => {
              setFormData({
                ...formData,
                INITDIR_North: e.target.value === "North" ? 1 : 0,
                INITDIR_South: e.target.value === "South" ? 1 : 0,
                INITDIR_West: e.target.value === "West" ? 1 : 0,
              });
            }}
          >
            <option value="">Select</option>
            <option value="North">North</option>
            <option value="South">South</option>
            <option value="West">West</option>
          </select>
        </div>
  
        {/* VEHTYPE */}
        <div className="mb-3">
          <label>Vehicle Type:</label>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="VEHTYPE_Automobile_Station_Wagon"
              checked={formData.VEHTYPE_Automobile_Station_Wagon === 1}
              onChange={() =>
                setFormData({
                  ...formData,
                  VEHTYPE_Automobile_Station_Wagon: 1,
                  VEHTYPE_Other: 0,
                })
              }
            />
            <label className="form-check-label">Automobile/Station Wagon</label>
          </div>
          <div className="form-check">
            <input
              type="radio"
              className="form-check-input"
              name="VEHTYPE_Other"
              checked={formData.VEHTYPE_Other === 1}
              onChange={() =>
                setFormData({
                  ...formData,
                  VEHTYPE_Automobile_Station_Wagon: 0,
                  VEHTYPE_Other: 1,
                })
              }
            />
            <label className="form-check-label">Other</label>
          </div>
        </div>
  
        {/* ROAD_CLASS_Major_Arterial */}
        <div className="mb-3">
          <label>Is the Road Classification Major Arterial?</label>
          <div className="form-check">
            <input
              type="checkbox"
              className="form-check-input"
              name="ROAD_CLASS_Major_Arterial"
              checked={formData.ROAD_CLASS_Major_Arterial === 1}
              onChange={handleChange}
            />
            <label className="form-check-label">Yes</label>
          </div>
        </div>
  
        <button type="submit" className="btn btn-primary">
          Submit
        </button>
      </form>
    </div>
  );  
}

export default App;