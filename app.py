"""
Flask backend API for bird classifier
"""
import os
import json
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.model import BirdClassifier
from config import NUM_CLASSES, CHECKPOINT_DIR, DEVICE

app = Flask(__name__)
CORS(app)

# Model and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdClassifier(num_classes=NUM_CLASSES).to(device)
model.eval()

# Bird class names - dynamically loaded from dataset
TRAIN_DIR = Path("data/Birds_25/train")

def get_bird_classes():
    """Load bird classes from actual dataset folders"""
    if TRAIN_DIR.exists():
        classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
        return classes
    else:
        # Fallback if dataset not found
        return [f"Class {i}" for i in range(NUM_CLASSES)]

BIRD_CLASSES = get_bird_classes()
print(f"[API] Loaded {len(BIRD_CLASSES)} bird classes from dataset")
print(f"[API] Classes: {BIRD_CLASSES[:5]}... (showing first 5)")

# Bird codenames for State of India's Birds website
BIRD_CODENAMES = {
    "Asian-Green-Bee-Eater": "grnbee3",
    "Brown-Headed-Barbet": "brhbar1",
    "Cattle-Egret": "categr",
    "Common-Kingfisher": "comkin1",
    "Common-Myna": "commyn",
    "Common-Rosefinch": "comros",
    "Common-Tailorbird": "comtai1",
    "Coppersmith-Barbet": "copbar1",
    "Forest-Wagtail": "forwag1",
    "Gray-Wagtail": "grywag",
    "Hoopoe": "hoopoe",
    "House-Crow": "houcro1",
    "Indian-Grey-Hornbill": "grehor1",
    "Indian-Peacock": "compea",
    "Indian-Pitta": "indpit1",
    "Indian-Roller": "indrol2",
    "Jungle-Babbler": "junbab2",
    "Northern-Lapwing": "norlap",
    "Red-Wattled-Lapwing": "rewlap1",
    "Ruddy-Shelduck": "rudshe",
    "Rufous-Treepie": "ruftre2",
    "Sarus-Crane": "sarcra1",
    "White-Breasted-Kingfisher": "whtkin2",
    "White-Breasted-Waterhen": "whbwat1",
    "White-Wagtail": "whiwag",
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.CenterCrop(456),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_best_model():
    """Load the best trained model"""
    best_model_path = Path(CHECKPOINT_DIR) / "best_model_stage2.pt"
    
    # Fall back to Stage 1 if Stage 2 not available
    if not best_model_path.exists():
        best_model_path = Path(CHECKPOINT_DIR) / "best_model_stage1.pt"
    
    if not best_model_path.exists():
        return False
    
    try:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[API] Loaded model: {best_model_path.name}")
        print(f"[API] Best accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": True
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict bird species from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Load and preprocess image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_idx = np.argmax(probabilities)
            predicted_confidence = probabilities[predicted_idx]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(probabilities)[::-1][:5]
        top_5_preds = [
            {
                "class": BIRD_CLASSES[idx],
                "confidence": float(probabilities[idx])
            }
            for idx in top_5_idx
        ]
        
        return jsonify({
            "predicted_class": BIRD_CLASSES[predicted_idx],
            "confidence": float(predicted_confidence),
            "top_5": top_5_preds
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of bird classes"""
    return jsonify({
        "classes": BIRD_CLASSES,
        "num_classes": len(BIRD_CLASSES)
    })

@app.route('/api/bird-info', methods=['GET'])
def bird_info():
    """Get bird information from State of India's Birds website"""
    bird_name = request.args.get('name', '')
    
    if not bird_name:
        return jsonify({"error": "Bird name required"}), 400
    
    # Get correct codename from mapping
    bird_code = BIRD_CODENAMES.get(bird_name)
    if not bird_code:
        print(f"[WARNING] Bird codename not found for: {bird_name}")
        return jsonify({
            "name": bird_name,
            "description": "Indian bird species",
            "facts": [
                f"• Species: {bird_name}",
                "• Region: India",
                "• Classification: Birds (Aves)"
            ]
        })
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Try to fetch from State of India's Birds website
        url = f"https://stateofindiasbirds.in/species/{bird_code}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract information from the page
                facts = []
                
                # Try to find description
                description = "Indian bird species"
                desc_elem = soup.find('p', class_='description') or soup.find('div', class_='description')
                if desc_elem:
                    description = desc_elem.get_text(strip=True)[:200]
                
                # Extract facts from the page
                facts.append(f"• Species: {bird_name}")
                facts.append("• Region: India")
                facts.append("• Classification: Birds (Aves)")
                
                # Try to extract conservation status
                status_elem = soup.find(text=re.compile(r'Conservation|Status', re.I))
                if status_elem:
                    facts.append(f"• {status_elem.get_text(strip=True)}")
                
                return jsonify({
                    "name": bird_name,
                    "description": description,
                    "codename": bird_code,
                    "facts": facts,
                    "source": "State of India's Birds"
                })
        except requests.exceptions.RequestException:
            # If website is unreachable, return generic info
            pass
        
        # Fallback response
        return jsonify({
            "name": bird_name,
            "description": "Indian bird species. Visit State of India's Birds for detailed information.",
            "codename": bird_code,
            "facts": [
                f"• Species: {bird_name}",
                "• Region: India",
                "• Classification: Birds (Aves)"
            ]
        })
    
    except Exception as e:
        print(f"[WARNING] Error fetching bird info: {e}")
        return jsonify({
            "name": bird_name,
            "description": "Indian bird species",
            "codename": bird_code,
            "facts": [
                f"• Species: {bird_name}",
                "• Region: India",
                "• Classification: Birds (Aves)"
            ]
        })

if __name__ == '__main__':
    print("="*70)
    print("BIRD CLASSIFIER API")
    print("="*70)
    print(f"[API] Device: {device}")
    print(f"[API] Loading model...")
    
    if load_best_model():
        print("[API] Model loaded successfully!")
        print("[API] Starting Flask server on http://localhost:5000\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("[ERROR] Could not load model. Train the model first!")
        print(f"[ERROR] Expected: {CHECKPOINT_DIR}/best_model_stage2.pt or best_model_stage1.pt")
