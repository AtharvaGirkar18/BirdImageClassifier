# React Frontend Setup

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Dev Server

```bash
npm run dev
```

Open: **http://localhost:3000**

### 3. Backend Must Be Running

In another terminal:

```cmd
C:\Users\handj\miniconda3\Scripts\activate.bat bird-img-gpu && python app.py
```

Backend runs on: **http://localhost:5000**

## Project Structure

```
frontend/
├── index.html              # Entry point
├── package.json            # Dependencies
├── vite.config.js          # Build config
├── src/
│   ├── main.jsx           # React entry
│   ├── main.css           # Global styles
│   ├── App.jsx            # Main component
│   ├── App.css            # App styles
│   └── components/
│       ├── ImageUpload.jsx    # Image uploader
│       ├── ImageUpload.css
│       ├── Predictions.jsx    # Predictions display
│       ├── Predictions.css
│       ├── BirdExplorer.jsx   # Bird info + LLM integration
│       └── BirdExplorer.css
```

## Components

### ImageUpload

- Drag & drop file upload
- Preview image before prediction
- Loading state

### Predictions

- Display top 5 predictions
- Confidence bars
- Click bird name to view details

### BirdExplorer

- Bird info display
- **LLM Integration Ready** ✨
- Placeholder for AI insights
- Click "Learn More" for future LLM queries

## LLM Integration (Ready to Build)

The `BirdExplorer` component has a placeholder for LLM integration:

```jsx
// In BirdExplorer.jsx - handleLearnMore() function
// Step 1: Create backend endpoint at /api/explore/{bird_name}
// Step 2: Call OpenAI/Claude/Local LLM API
// Step 3: Stream response to user
```

### Future: Add Your LLM

**Option 1: OpenAI API**

```jsx
const response = await fetch("http://localhost:5000/api/explore", {
  method: "POST",
  body: JSON.stringify({
    bird_name: selectedBird,
    query: "Tell me about this bird",
  }),
});
```

**Option 2: Local LLM (Ollama)**

```jsx
const response = await fetch("http://localhost:11434/api/generate", {
  method: "POST",
  body: JSON.stringify({
    model: "mistral",
    prompt: `Tell me about ${selectedBird}`,
  }),
});
```

**Option 3: Anthropic Claude**

```jsx
const response = await fetch("http://localhost:5000/api/claude", {
  method: "POST",
  body: JSON.stringify({
    bird_name: selectedBird,
    max_tokens: 500,
  }),
});
```

## Backend Integration Points

### 1. Prediction Endpoint (Already Working)

```
POST /api/predict
- Accepts image upload
- Returns top 5 predictions
```

### 2. Bird Exploration Endpoint (To Build)

```
POST /api/explore
- Input: bird_name
- Output: AI-powered insights
- Connect to LLM of choice
```

### 3. Chat Endpoint (Future)

```
POST /api/chat
- Input: user message about bird
- Output: LLM response with context
```

## Build for Production

### 1. Build Static Files

```bash
npm run build
```

Creates optimized files in `dist/` folder

### 2. Deploy Frontend

- Upload `dist/` to Netlify, Vercel, or any static host
- Update API URL in code for production

### 3. Deploy Backend

- Use Flask production server (Gunicorn)
- Deploy to AWS, Google Cloud, or Heroku

## Development Tips

### Hot Reload

Edit any component - browser auto-updates without refresh

### Debug in Browser

- Open DevTools (F12)
- React Developer Tools extension recommended
- Network tab to see API calls

### Test Prediction Flow

1. Start both frontend (3000) and backend (5000)
2. Upload bird image
3. See predictions appear
4. Click bird name to test BirdExplorer

## Performance

- **Vite**: Fast bundling, HMR (hot module reload)
- **React 18**: Latest features, concurrent rendering
- **CSS modules**: Scoped styles, no conflicts
- **API caching**: Add later as needed

## Common Issues

### "Could not connect to API"

- Backend not running at localhost:5000
- Check: `python app.py` is executing

### CORS errors

- Backend `app.py` already has CORS enabled
- Check both servers are running

### Image won't upload

- File too large (check 10MB limit in code)
- Wrong file format (use PNG/JPG/GIF)
- Check browser console for errors

## Next Steps

1. ✅ Frontend running
2. ✅ Predictions working
3. **🔜 Integrate LLM for bird insights**
4. **🔜 Add chat features**
5. **🔜 Deploy to production**

## Resources

- [React Docs](https://react.dev)
- [Vite Guide](https://vitejs.dev)
- [Axios Docs](https://axios-http.com)
- [OpenAI API](https://platform.openai.com)
- [Anthropic Claude](https://www.anthropic.com)
