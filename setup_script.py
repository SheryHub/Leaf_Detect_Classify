#!/usr/bin/env python3
"""
Setup and run script for Leaf Disease Detection Application
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_models():
    """Check if model files exist"""
    models_dir = Path("models")
    yolo_model = models_dir / "best.pt"
    resnet_model = models_dir / "plant_disease_resnet_model.pth"
    
    if not models_dir.exists():
        print("❌ 'models' directory not found")
        print("Please create a 'models' directory and place your model files there")
        return False
    
    if not yolo_model.exists():
        print("❌ YOLO model 'best.pt' not found in models directory")
        return False
    else:
        print("✅ YOLO model found")
    
    if not resnet_model.exists():
        print("❌ ResNet model 'plant_disease_resnet_model.pth' not found in models directory")
        return False
    else:
        print("✅ ResNet model found")
    
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directory_structure():
    """Create necessary directories"""
    directories = ["models", "static", "uploads"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory '{directory}' ready")

def check_file_structure():
    """Check if all necessary files exist"""
    required_files = [
        "app.py",
        "requirements.txt",
        "index.html"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file '{file}' not found")
            return False
        else:
            print(f"✅ File '{file}' found")
    
    return True

def main():
    print("🌿 Leaf Disease Detection Application Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check file structure
    if not check_file_structure():
        print("\n❌ Setup failed: Missing required files")
        print("Please make sure you have all the necessary files in your project directory")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Check models
    if not check_models():
        print("\n❌ Setup failed: Model files missing")
        print("Please place your trained models in the 'models' directory:")
        print("  - models/best.pt (YOLO model)")
        print("  - models/plant_disease_resnet_model.pth (ResNet model)")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n" + "=" * 50)
    print("📋 Next Steps:")
    print("1. Run the Flask server: python app.py")
    print("2. Open index.html in your web browser")
    print("3. Or serve the HTML file using: python -m http.server 8000")
    print("4. Then visit: http://localhost:8000")
    print("\n💡 Tips:")
    print("- Make sure both the Flask server (port 5000) and web server are running")
    print("- The web app will communicate with the Flask API for model inference")
    print("- Check the browser console for any connection errors")

if __name__ == "__main__":
    main()