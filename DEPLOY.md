# Bird Classifier - Frontend & Backend Setup

## Overview

- **Backend**: Flask API (`app.py`) - runs on `http://localhost:5000`
- **Frontend**: HTML/CSS/JS (`index.html`) - plain web interface
- **Model**: Loads best trained model and serves predictions

## Backend Setup

### 1. Install Dependencies

```cmd
conda run -n bird-img-gpu pip install flask flask-cors
```

Or if dependencies already in environment:

```cmd
C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu && pip install flask flask-cors
```

### 2. Run Backend

```cmd
C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu && python app.py
```

Output should show:

```
==================================================
BIRD CLASSIFIER API
==================================================
[API] Device: cuda
[API] Loading model...
[API] Model loaded successfully!
[API] Starting Flask server on http://localhost:5000
```

## Frontend Setup

### 1. Open in Browser

Once backend is running, open `index.html` in any browser:

- **Option A**: Right-click `index.html` → Open with browser
- **Option B**: Drag `index.html` onto browser window
- **Option C**: Type in browser: `file:///C:/Users/handj/bird-image-classifier/index.html`

### 2. Use the Interface

1. Click upload area or drag an image
2. Click "Predict Species"
3. See prediction with confidence score and top 5 results

## How It Works

### Backend (`app.py`)

1. Loads best trained model from `checkpoints/best_model_stage2.pt` or `best_model_stage1.pt`
2. Listens for image uploads on `/api/predict`
3. Preprocesses image (resize, normalize)
4. Runs inference on GPU (if available)
5. Returns top 5 predictions with confidence scores

### Frontend (`index.html`)

1. Pure HTML/CSS/JavaScript - no build needed
2. Sends image to backend via `fetch()` API
3. Shows real-time predictions with confidence bars
4. Fully responsive - works on mobile/tablet/desktop

## API Endpoints

### Health Check

```
GET http://localhost:5000/api/health
```

### Get Bird Classes

```
GET http://localhost:5000/api/classes
```

### Predict

```
POST http://localhost:5000/api/predict
Body: FormData with 'image' file
Response: {
    "predicted_class": "Indian Peafowl",
    "confidence": 0.95,
    "top_5": [
        {"class": "Indian Peafowl", "confidence": 0.95},
        {"class": "Common Peacock", "confidence": 0.03},
        ...
    ]
}
```

## Troubleshooting

### "Could not connect to API"

- Make sure backend is running: `python app.py`
- Check localhost:5000/api/health from browser

### "Model not found"

- Train the model first: `python run_training.py`
- Or resume training: `python resume_training.py`
- Model must be saved as `best_model_stage1.pt` or `best_model_stage2.pt` in `checkpoints/`

### Slow predictions

- CPU inference is slow - ensure GPU is available
- Wait for model to load on startup (30-60 seconds)

### CORS errors

- Backend has CORS enabled in `app.py`
- Should work with frontend on file:// protocol

## Next Steps

1. **Improve accuracy**: Continue training Stage 2 for more epochs
2. **Deploy online**: Use Heroku, AWS, or Google Cloud
3. **Add more features**: Batch predictions, image gallery, API documentation
4. **Mobile app**: Convert to React Native or Flutter

## Files

```
bird-image-classifier/
├── app.py              # Flask backend
├── index.html          # Web frontend
├── backend_requirements.txt
└── DEPLOY.md          # This file
```
