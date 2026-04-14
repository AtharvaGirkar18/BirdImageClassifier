import { useState, useRef } from "react";
import "./ImageUpload.css";

export default function ImageUpload({ onPredict, loading }) {
  const [preview, setPreview] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
      onPredict(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  };

  return (
    <div className="image-upload">
      <div
        className={`upload-area ${isDragOver ? "dragover" : ""}`}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="upload-icon">📸</div>
        <div className="upload-text">Click to upload or drag and drop</div>
        <div className="upload-hint">PNG, JPG, GIF up to 10MB</div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          style={{ display: "none" }}
        />
      </div>

      {preview && (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
          {loading && (
            <div className="preview-overlay">
              <div className="small-spinner"></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
