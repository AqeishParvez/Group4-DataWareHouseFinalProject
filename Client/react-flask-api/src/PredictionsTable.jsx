import React, { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

function PredictionsTable() {
  const [predictions, setPredictions] = useState([]);

  // Fetch predictions from the backend
  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/predictions");
        const data = await response.json();
        setPredictions(data);
      } catch (error) {
        console.error("Error fetching predictions:", error);
      }
    };

    fetchPredictions();
  }, []);

  // Consolidate responses
  const consolidateResponses = (input) => {
    const responses = [];

    // Initial Direction
    if (input.INITDIR_North) responses.push("Initial Direction: North");
    else if (input.INITDIR_South) responses.push("Initial Direction: South");
    else if (input.INITDIR_West) responses.push("Initial Direction: West");
    else responses.push("Initial Direction: Unknown");

    // Involvement Type
    if (input.INVTYPE_Driver) responses.push("Involvement Type: Driver");
    else if (input.INVTYPE_Passenger) responses.push("Involvement Type: Passenger");
    else if (input.INVTYPE_Pedestrian) responses.push("Involvement Type: Pedestrian");
    else responses.push("Involvement Type: Unknown");

    // Driver Condition
    if (input.DRIVCOND_Normal) responses.push("Driver Condition: Normal");
    else if (input.DRIVCOND_Unknown) responses.push("Driver Condition: Unknown");
    else responses.push("Driver Condition: Other");

    // Other responses
    responses.push(`Was the vehicle going ahead?: ${input.MANOEUVER_Going_Ahead ? "Yes" : "No"}`);
    responses.push(`Was the driver driving properly?: ${input.DRIVACT_Driving_Properly ? "Yes" : "No"}`);
    responses.push(`Was the collision with a pedestrian?: ${input.IMPACTYPE_Pedestrian_Collisions ? "Yes" : "No"}`);
    responses.push(`Was the road classified as 'Major Arterial'?: ${input.ROAD_CLASS_Major_Arterial ? "Yes" : "No"}`);
    responses.push(`Hour of the incident?: ${input.HOUR}`);
    responses.push(`Year of the incident?: ${input.YEAR}`);
    responses.push(`Was the vehicle an automobile or station wagon?: ${input.VEHTYPE_Automobile_Station_Wagon ? "Yes" : "No"}`);
    responses.push(`Was the vehicle type 'Other'?: ${input.VEHTYPE_Other ? "Yes" : "No"}`);
    responses.push(`Was the age unknown?: ${input.INVAGE_unknown ? "Yes" : "No"}`);

    return responses.join("\n");
  };

  // Format prediction outcome
  const formatPrediction = (prediction) => {
    return prediction === 1 ? (
      <span style={{ color: "red", fontWeight: "bold" }}>Fatal</span>
    ) : (
      <span style={{ color: "blue", fontWeight: "bold" }}>Non-Fatal</span>
    );
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4">Prediction History</h1>
      <table className="table table-striped table-bordered">
        <thead className="thead-dark">
          <tr>
            <th>Timestamp</th>
            <th>Prediction</th>
            <th>Probability</th>
            <th>Responses</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((prediction, index) => (
            <tr key={index}>
              {/* Format timestamp */}
              <td>{new Date(prediction.timestamp).toLocaleString()}</td>

              {/* Format prediction */}
              <td>{formatPrediction(prediction.prediction)}</td>

              {/* Keep probability as is */}
              <td>{prediction.probability.toFixed(2)}</td>

              {/* Display consolidated responses */}
              <td>
                <pre>{consolidateResponses(prediction.input)}</pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionsTable;