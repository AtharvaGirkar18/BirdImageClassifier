# 🚀 Bird Classifier - Production Deployment

Model Accuracy: **82.48%** ✅

## Quick Start (3 Steps)

### Step 1: Install Frontend Dependencies (One-time)

```cmd
cd frontend
npm install
```

### Step 2: Start Backend (Terminal 1)

```cmd
C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu && python app.py
```

Expected output:

```
==================================================
BIRD CLASSIFIER API
==================================================
[API] Device: cuda
[API] Loading model...
[API] Model loaded successfully!
[API] Starting Flask server on http://localhost:5000
```

### Step 3: Start Frontend (Terminal 2)

```cmd
cd frontend
npm run dev
```

Expected output:

```
  VITE v5.0.0 ready in 234 ms

  ➜  Local:   http://localhost:3000/
```

**Open in Browser: http://localhost:3000**

---

## How to Use

1. **Upload Image**: Drag & drop or click to upload bird image
2. **Get Prediction**: Model predicts bird species with confidence
3. **View Top 5**: See all predictions ranked by confidence
4. **Learn More**: Click "Learn More" button (LLM integration ready)

---

## API Endpoints

### Health Check

```bash
curl http://localhost:5000/api/health
```

### Get Bird Classes

```bash
curl http://localhost:5000/api/classes
```

### Predict (Upload Image)

```bash
curl -X POST -F "image=@bird.jpg" http://localhost:5000/api/predict
```

Response:

```json
{
  "predicted_class": "Indian Peafowl",
  "confidence": 0.8248,
  "top_5": [
    {"class": "Indian Peafowl", "confidence": 0.82},
    {"class": "Common Peafowl", "confidence": 0.10},
    ...
  ]
}
```

---

## Troubleshooting

### "Module not found" when running backend

```cmd
C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu
pip install flask flask-cors pytorch torchvision
python app.py
```

### "npm: command not found"

- Install Node.js: https://nodejs.org/
- Restart terminal and try again

### "Could not connect to API" in frontend

- Ensure backend is running on http://localhost:5000
- Check firewall isn't blocking port 5000
- Try: http://localhost:5000/api/health in browser

### "Model not found" error

- Model should be at: `checkpoints/best_model_stage2.pt`
- Or: `checkpoints/best_model_stage1.pt`
- Train first if missing: `python run_training.py`

### Image upload not working

- Check file size < 10MB
- Use PNG, JPG, or GIF format
- Check browser console (F12) for error messages

---

## Production Deployment

### Option 1: Deploy Anywhere

1. Backend: Heroku, AWS Lambda, Google Cloud Run
2. Frontend: Netlify, Vercel, GitHub Pages

### Option 2: Docker Containerization (Coming Soon)

```dockerfile
# Backend Dockerfile
FROM python:3.11
RUN pip install flask torch torchvision
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
```

### Option 3: Advanced (Gunicorn + Nginx)

```bash
# Production backend server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Next Steps

1. ✅ **Model Training**: Stage 2 Epoch 0 complete (82.48%)
2. ✅ **Backend Ready**: Flask API running
3. ✅ **Frontend Ready**: React + Vite
4. **🔜 LLM Integration**: Add bird insights (OpenAI/Claude)
5. **🔜 Production**: Deploy to cloud

---

## LLM Integration (When Ready)

To add AI-powered bird insights:

### Backend (`app.py`):

```python
@app.route('/api/explore', methods=['POST'])
def explore_bird():
    bird_name = request.json.get('bird_name')

    # Option 1: OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Tell me about {bird_name}"
        }]
    )
    return jsonify({"insights": response.choices[0].message.content})
```

### Frontend (`BirdExplorer.jsx`):

```jsx
const handleLearnMore = async () => {
  const response = await fetch("/api/explore", {
    method: "POST",
    body: JSON.stringify({ bird_name: selectedBird }),
  });
  const data = await response.json();
  setLlmResponse(data.insights);
};
```

---

## Model Stats

| Metric             | Value              |
| ------------------ | ------------------ |
| Architecture       | EfficientNet-B4    |
| Parameters         | 18.6M              |
| Training Samples   | 30,000             |
| Validation Samples | 7,500              |
| Classes            | 25 Indian Birds    |
| Best Accuracy      | 82.48%             |
| Stage 1 Accuracy   | 82.20%             |
| Training Time      | ~1 hour            |
| GPU                | NVIDIA GTX 1650 Ti |

---

## Support

For issues:

1. Check terminal output for error messages
2. Review troubleshooting section above
3. Check model checkpoints folder
4. Verify ports 5000 (backend) and 3000 (frontend) are available

---

## Files

```
bird-image-classifier/
├── app.py                  # Flask backend API
├── deploy.py               # Deployment helper
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── ImageUpload.jsx
│       │   ├── Predictions.jsx
│       │   └── BirdExplorer.jsx
│       └── ...
├── checkpoints/
│   └── best_model_stage2.pt    # Your trained model (82.48%)
└── ...
```

Enjoy! 🦅
