import "./Predictions.css";

export default function Predictions({ predictions, onClear, onSelectBird }) {
  const confidence = Math.round(predictions.confidence * 100);

  return (
    <div className="predictions">
      <div className="main-prediction">
        <div className="predicted-class">{predictions.predicted_class}</div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${confidence}%` }}
          ></div>
        </div>
        <div className="confidence-text">Confidence: {confidence}%</div>
      </div>

      <div className="top-predictions">
        <h3>Top 5 Predictions</h3>
        <div className="predictions-list">
          {predictions.top_5.map((pred, idx) => (
            <div
              key={idx}
              className="prediction-item"
              onClick={() => onSelectBird(pred.class)}
            >
              <div className="prediction-rank">#{idx + 1}</div>
              <div className="prediction-name">{pred.class}</div>
              <div className="prediction-confidence">
                {Math.round(pred.confidence * 100)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      <button className="btn-clear" onClick={onClear}>
        Clear & Upload New
      </button>
    </div>
  );
}
