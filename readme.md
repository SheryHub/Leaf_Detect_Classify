# 🌿 Leaf Disease Detection Web Application

A complete web application for detecting leaves and classifying plant diseases using trained YOLO and ResNet models.

## 📁 Project Structure

```
leaf/
├── models/
│   ├── best.pt                          # Your trained YOLO model
│   └── plant_disease_resnet_model.pth    # Your trained ResNet model
├── app.py                               # Flask server (backend)
├── index.html                           # Web application (frontend)
├── requirements.txt                     # Python dependencies
├── setup_and_run.py                    # Setup script
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Setup
Run the setup script to check everything and install dependencies:

```bash
python setup_and_run.py
```

### 2. Start the Flask Server
```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Open the Web Application
#### Option A: Direct file access
Open `index.html` directly in your web browser

#### Option B: Local web server (recommended)
```bash
python -m http.server 8000
```
Then visit `http://localhost:8000`

## 🎯 Features

### 📷 Live Camera Capture
- Access device camera (desktop/mobile)
- Real-time video feed
- Capture high-quality images

### 🔍 Leaf Detection (YOLO)
- Detect and locate leaves in images
- Draw bounding boxes around detected leaves
- Show confidence scores for each detection

### 🦠 Disease Classification (ResNet)
- Classify plant diseases from captured images
- Support for 17 different plant disease classes:
  - **Corn**: Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight
  - **Potato**: Early Blight, Healthy, Late Blight
  - **Rice**: Brown Spot, Healthy, Leaf Blast, Neck Blast
  - **Wheat**: Brown Rust, Healthy, Yellow Rust
  - **Sugarcane**: Red Rot, Healthy, Bacterial Blight

### 📱 Responsive Design
- Works on desktop and mobile devices
- Touch-friendly interface
- Optimized for various screen sizes

## 🛠️ Technical Details

### Backend (Flask)
- **Framework**: Flask with CORS support
- **Models**: PyTorch YOLO and ResNet
- **Image Processing**: PIL/Pillow
- **API Endpoints**:
  - `GET /health` - Server health check
  - `POST /detect` - Leaf detection
  - `POST /classify` - Disease classification
  - `POST /analyze` - Complete analysis

### Frontend (HTML/CSS/JavaScript)
- **Camera Access**: WebRTC getUserMedia API
- **Image Capture**: HTML5 Canvas
- **UI Framework**: Vanilla JavaScript with modern CSS
- **Styling**: Gradient backgrounds, animations, responsive design

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Camera**: Web camera or mobile camera for live capture

### Python Dependencies
- `torch>=2.0.0` - PyTorch framework
- `torchvision>=0.15.0` - Computer vision utilities
- `ultralytics>=8.0.0` - YOLO implementation
- `flask>=2.3.0` - Web framework
- `flask-cors>=4.0.0` - Cross-origin resource sharing
- `pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing
- `opencv-python>=4.5.0` - Computer vision

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Leaf Detection
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/detect
```

### Disease Classification
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/classify
```

### Complete Analysis
```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/analyze
```

## 🎨 Customization

### Adding New Disease Classes
1. Update `CLASS_LABELS` in `app.py`
2. Retrain your ResNet model with new classes
3. Update the model file

### Changing Model Paths
Modify the model paths in `app.py`:
```python
yolo_path = os.path.join('models', 'your_yolo_model.pt')
resnet_path = os.path.join('models', 'your_resnet_model.pth')
```

### UI Customization
Edit the CSS in `index.html` to change:
- Colors and gradients
- Button styles
- Layout and spacing
- Animations and effects

## 🐛 Troubleshooting

### Common Issues

**1. Models not loading**
- Ensure model files are in the `models/` directory
- Check file names match exactly: `best.pt` and `plant_disease_resnet_model.pth`
- Verify models are compatible with your PyTorch version

**2. Camera not working**
- Allow camera permissions in your browser
- Use HTTPS for production deployment
- Check browser compatibility (Chrome/Firefox recommended)

**3. Server connection errors**
- Ensure Flask server is running on port 5000
- Check CORS settings if accessing from different domains
- Verify firewall settings

**4. Memory issues**
- Reduce image resolution for processing
- Use CPU instead of GPU if CUDA memory is limited
- Close other applications to free up RAM

### Error Messages

- **"Pipeline not initialized"**: Models failed to load
- **"No image provided"**: Image upload failed
- **"Server not connected"**: Flask server is not running
- **"Camera not accessible"**: Browser permissions or hardware issue

## 🚀 Deployment Options

### Local Development
- Run Flask server locally
- Serve HTML file via local web server
- Access via localhost

### Production Deployment
- **Backend**: Deploy Flask app to Heroku, AWS, or DigitalOcean
- **Frontend**: Deploy to Netlify, Vercel, or GitHub Pages
- **Models**: Use cloud storage for large model files

### Docker Deployment
Create a `Dockerfile` for containerized deployment:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 📊 Performance Notes

- **YOLO Inference**: ~100-500ms per image
- **ResNet Inference**: ~50-200ms per image
- **Total Processing**: ~200-700ms per image
- **Memory Usage**: ~2-4GB with models loaded

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **PyTorch** for the deep learning framework
- **Ultralytics** for the YOLO implementation
- **Flask** for the web framework
- **Your trained models** for making this application possible

---

**Happy detecting! 🌿🔬**