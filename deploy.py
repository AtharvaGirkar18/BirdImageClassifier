#!/usr/bin/env python3
"""
Complete deployment setup for Bird Classifier
Run all services at once
"""
import os
import subprocess
import sys
import time
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def main():
    root = Path(__file__).parent
    
    print_header("🦅 BIRD CLASSIFIER - DEPLOYMENT")
    
    # Check backend requirements
    print("[1/3] Checking backend requirements...")
    backend_ok = True
    try:
        import flask
        import torch
        import torchvision
        print("✓ Backend dependencies OK (Flask, PyTorch installed)")
    except ImportError as e:
        print(f"✗ Missing: {e}")
        backend_ok = False
    
    # Check model
    print("\n[2/3] Checking trained model...")
    model_paths = [
        root / "checkpoints" / "best_model_stage2.pt",
        root / "checkpoints" / "best_model_stage1.pt"
    ]
    
    model_found = False
    for path in model_paths:
        if path.exists():
            print(f"✓ Found model: {path.name}")
            model_found = True
            break
    
    if not model_found:
        print("✗ No trained model found!")
        print("  Train the model first: python run_training.py")
        sys.exit(1)
    
    # Check frontend
    print("\n[3/3] Checking frontend setup...")
    frontend_dir = root / "frontend"
    if (frontend_dir / "package.json").exists():
        print("✓ React frontend found")
    else:
        print("✗ Frontend not found")
    
    print_header("🚀 DEPLOYMENT OPTIONS")
    
    print("Option 1: Run Backend Only")
    print("  python app.py")
    print("  Then open frontend: frontend/index.html in browser")
    
    print("\nOption 2: Run Frontend (Dev Server)")
    print("  cd frontend")
    print("  npm install  # First time only")
    print("  npm run dev")
    print("  Then run backend in another terminal")
    
    print("\nOption 3: Full Stack (This Script)")
    print("  Running both backend and frontend...")
    
    choice = input("\nSelect option (1-3) or 'q' to quit: ").strip().lower()
    
    if choice == 'q':
        print("Exiting.")
        sys.exit(0)
    
    elif choice == '1':
        print_header("Starting Backend Only")
        print("Backend running on: http://localhost:5000")
        print("Open frontend: file:///C:/Users/handj/bird-image-classifier/index.html")
        print("Press Ctrl+C to stop\n")
        os.system("python app.py")
    
    elif choice == '2':
        print_header("Starting Frontend (Dev Server)")
        print("Frontend running on: http://localhost:3000")
        print("Make sure backend is running: python app.py")
        print("Press Ctrl+C to stop\n")
        os.chdir(frontend_dir)
        os.system("npm run dev")
    
    elif choice == '3':
        print_header("Starting Full Stack Deployment")
        print("\nStarting Backend (Terminal 1)...")
        print("Backend: http://localhost:5000")
        print("Frontend: http://localhost:3000")
        print("\nPress Ctrl+C in either terminal to stop\n")
        
        # Try to start backend in background
        import platform
        try:
            if platform.system() == 'Windows':
                subprocess.Popen([sys.executable, "app.py"], cwd=root)
                time.sleep(3)
            
            print("\nStarting Frontend (Terminal 2)...")
            os.chdir(frontend_dir)
            os.system("npm run dev")
        except Exception as e:
            print(f"Error: {e}")
            print("\nStart backend manually: python app.py")
            print("Then start frontend: cd frontend && npm run dev")
    
    else:
        print("Invalid option")
        sys.exit(1)

if __name__ == "__main__":
    main()
