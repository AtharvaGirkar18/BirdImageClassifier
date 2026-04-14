import { useState } from "react";
import ImageUpload from "./components/ImageUpload";
import Predictions from "./components/Predictions";
import BirdExplorer from "./components/BirdExplorer";
import "./App.css";

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedBird, setSelectedBird] = useState(null);

  const handlePredict = async (image) => {
    setSelectedImage(image);
    setLoading(true);
    setError(null);
    setPredictions(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Prediction failed");

      const data = await response.json();
      setPredictions(data);
      setSelectedBird(data.predicted_class);
    } catch (err) {
      setError(err.message || "Could not connect to API");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedImage(null);
    setPredictions(null);
    setSelectedBird(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🦅 Bird Species Classifier</h1>
        <p>Identify Indian birds with AI-powered vision</p>
      </header>

      <div className="app-container">
        <main className="app-main">
          <ImageUpload onPredict={handlePredict} loading={loading} />

          {error && (
            <div className="error-banner">
              <span>⚠️ {error}</span>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Analyzing image...</p>
            </div>
          )}

          {predictions && (
            <Predictions
              predictions={predictions}
              onClear={handleClear}
              onSelectBird={setSelectedBird}
            />
          )}
        </main>

        <aside className="app-sidebar">
          <BirdExplorer selectedBird={selectedBird} />
        </aside>
      </div>
    </div>
  );
}
