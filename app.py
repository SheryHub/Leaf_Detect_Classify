import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np

warnings.filterwarnings('ignore')

# Class labels for plant diseases
CLASS_LABELS = [
    'Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Healthy', 'Corn_Northern_Leaf_Blight',
    'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight',
    'Rice_Brown_Spot', 'Rice_Healthy', 'Rice_Leaf_Blast', 'Rice_Neck_Blast',
    'Wheat_Brown_Rust', 'Wheat_Healthy', 'Wheat_Yellow_Rust',
    'Sugarcane_Red_Rot', 'Sugarcane_Healthy', 'Sugarcane_Bacterial_Blight'
]

class ResNetPlantDisease(nn.Module):
    """ResNet model for plant disease classification"""
    
    def __init__(self, num_classes=17, model_name='resnet50', pretrained=True):
        super(ResNetPlantDisease, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError("Only resnet50 supported in this demo")
        
        # Replace the classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class LeafDiseaseDetectionPipeline:
    def __init__(self, yolo_model_path, resnet_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load YOLO model for leaf detection
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            self.yolo_model = None
        
        # Load ResNet model for disease classification
        try:
            self.resnet_model = ResNetPlantDisease(num_classes=len(CLASS_LABELS))
            if os.path.exists(resnet_model_path):
                state_dict = torch.load(resnet_model_path, map_location=self.device)
                self.resnet_model.load_state_dict(state_dict)
                print("✓ ResNet model loaded successfully")
            else:
                print("✗ ResNet model file not found, using untrained model")
            
            self.resnet_model.to(self.device)
            self.resnet_model.eval()
        except Exception as e:
            print(f"✗ Error loading ResNet model: {e}")
            self.resnet_model = None
        
        # Image transforms for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_leaves(self, image):
        """Detect leaves using YOLO model"""
        if self.yolo_model is None:
            return {"success": False, "error": "YOLO model not loaded"}
        
        try:
            # Run YOLO inference
            results = self.yolo_model(image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Convert to width/height format
                        bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                        
                        detections.append({
                            "class": f"Leaf",  # or use class names if available
                            "confidence": float(confidence),
                            "bbox": bbox
                        })
            
            return {
                "success": True,
                "detections": detections,
                "count": len(detections)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def classify_disease(self, image):
        """Classify plant disease using ResNet model"""
        if self.resnet_model is None:
            return {"success": False, "error": "ResNet model not loaded"}
        
        try:
            # Preprocess image
            if isinstance(image, str):  # if base64 string
                image = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.resnet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top 5 predictions
                top_probs, top_indices = torch.topk(probabilities, 5)
                
                predictions = []
                for i in range(len(top_probs)):
                    class_idx = top_indices[i].item()
                    confidence = top_probs[i].item()
                    class_name = CLASS_LABELS[class_idx]
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "class_id": class_idx
                    })
                
                return {
                    "success": True,
                    "predictions": predictions,
                    "top_prediction": predictions[0] if predictions else None
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the pipeline
pipeline = None

def initialize_pipeline():
    global pipeline
    try:
        yolo_path = os.path.join('models', 'best.pt')
        resnet_path = os.path.join('models', 'plant_disease_resnet_model.pth')
        
        pipeline = LeafDiseaseDetectionPipeline(yolo_path, resnet_path)
        print("Pipeline initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if pipeline is None:
        return jsonify({"status": "error", "message": "Pipeline not initialized"}), 500
    
    return jsonify({
        "status": "healthy",
        "yolo_loaded": pipeline.yolo_model is not None,
        "resnet_loaded": pipeline.resnet_model is not None,
        "device": pipeline.device
    })

@app.route('/detect', methods=['POST'])
def detect_leaves():
    """Endpoint for leaf detection"""
    if pipeline is None:
        return jsonify({"success": False, "error": "Pipeline not initialized"}), 500
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No image selected"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run detection
        results = pipeline.detect_leaves(image)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_disease():
    """Endpoint for disease classification"""
    if pipeline is None:
        return jsonify({"success": False, "error": "Pipeline not initialized"}), 500
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No image selected"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run classification
        results = pipeline.classify_disease(image)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_complete():
    """Endpoint for complete analysis (detection + classification)"""
    if pipeline is None:
        return jsonify({"success": False, "error": "Pipeline not initialized"}), 500
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No image selected"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run both detection and classification
        detection_results = pipeline.detect_leaves(image)
        classification_results = pipeline.classify_disease(image)
        
        return jsonify({
            "success": True,
            "detection": detection_results,
            "classification": classification_results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Leaf Disease Detection API",
        "endpoints": {
            "/health": "Check server health",
            "/detect": "POST - Detect leaves in image",
            "/classify": "POST - Classify plant disease",
            "/analyze": "POST - Complete analysis (detection + classification)"
        }
    })

if __name__ == '__main__':
    print("Starting Leaf Disease Detection Server...")
    print("Initializing models...")
    
    if initialize_pipeline():
        print("✓ Server ready!")
        print("Available endpoints:")
        print("  - GET  /health     - Health check")
        print("  - POST /detect     - Leaf detection")
        print("  - POST /classify   - Disease classification")
        print("  - POST /analyze    - Complete analysis")
        print("\nStarting Flask server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("✗ Failed to initialize pipeline. Please check your model files.")
        print("Make sure you have:")
        print("  - models/best.pt (YOLO model)")
        print("  - models/plant_disease_resnet_model.pth (ResNet model)")
                