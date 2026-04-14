import { useState, useEffect } from "react";
import "./BirdExplorer.css";

export default function BirdExplorer({ selectedBird }) {
  const [birdInfo, setBirdInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Bird-specific codenames for linking to State of India's Birds
  const getBirdLink = (birdName) => {
    // This will be provided by the backend instead
    return null;
  };

  useEffect(() => {
    if (selectedBird) {
      setLoading(true);
      setError(null);
      fetchBirdInfo(selectedBird);
    } else {
      setBirdInfo(null);
      setError(null);
    }
  }, [selectedBird]);

  const fetchBirdInfo = async (birdName) => {
    try {
      // Call backend to fetch bird info from State of India's Birds website
      const response = await fetch(
        `/api/bird-info?name=${encodeURIComponent(birdName)}`,
      );

      if (response.ok) {
        const data = await response.json();
        setBirdInfo(data);
      } else {
        // Fallback to basic info if API fails
        setBirdInfo({
          name: birdName,
          description: "Indian bird species",
          facts: [
            "• Species: " + birdName,
            "• Classification: Birds (Aves)",
            "• Region: India",
          ],
        });
      }
    } catch (err) {
      console.error("Error fetching bird info:", err);
      // Fallback if fetch fails
      setBirdInfo({
        name: birdName,
        description: "Indian bird species",
        facts: [
          "• Species: " + birdName,
          "• Classification: Birds (Aves)",
          "• Region: India",
        ],
      });
    } finally {
      setLoading(false);
    }
  };

  if (!selectedBird) {
    return (
      <div className="bird-explorer">
        <div className="empty-state">
          <div className="empty-icon">🔍</div>
          <p>Upload an image to learn about the bird species</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bird-explorer">
      <h2>🦅 {birdInfo?.name}</h2>

      {loading ? (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Loading bird information...</p>
        </div>
      ) : error ? (
        <div className="error-state">
          <p>Unable to fetch bird information</p>
        </div>
      ) : (
        <div className="bird-info">
          <p className="description">{birdInfo?.description}</p>

          {birdInfo?.facts && (
            <div className="facts">
              <h4>Information</h4>
              {birdInfo.facts.map((fact, idx) => (
                <p key={idx}>{fact}</p>
              ))}
            </div>
          )}

          {birdInfo?.codename && (
            <a
              href={`https://stateofindiasbirds.in/species/${birdInfo.codename}`}
              target="_blank"
              rel="noopener noreferrer"
              className="link-to-source"
            >
              📖 Learn More on State of India&apos;s Birds
            </a>
          )}
        </div>
      )}
    </div>
  );
}
