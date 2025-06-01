# Video Analysis System - Project Structure

## Required Folder Structure

```
video_analysis_system/
│
├── main_app.py                 # Main Flask application
├── requirements.txt            # Python dependencies
│
├── templates/                  # HTML templates folder
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Home page with upload form
│   ├── results.html           # Analysis results display
│   ├── face_match.html        # Face matching interface
│   ├── plate_search.html      # License plate search
│   └── alerts.html            # System alerts page
│
├── static/                    # Static files
│   ├── uploads/               # Uploaded videos (auto-created)
│   └── frames/                # Extracted frames (auto-created)
│
├── database/                  # SQLite database (auto-created)
│   └── metadata.db           # Main database file
│
└── models/                    # AI models (auto-created)
    └── yolov8n.pt            # YOLO model (auto-downloaded)
```

## Installation Instructions

1. **Create the project directory:**
   ```bash
   mkdir video_analysis_system
   cd video_analysis_system
   ```

2. **Create the templates folder:**
   ```bash
   mkdir templates
   ```

3. **Save all HTML files to the templates folder:**
   - Save each HTML template in the `templates/` directory
   - Make sure file names match exactly: `base.html`, `index.html`, etc.

4. **Install Python dependencies:**
   ```bash
   pip install flask opencv-python sqlite3 numpy ultralytics easyocr facenet-pytorch torch deep-sort-realtime werkzeug pathlib
   ```

5. **Run the application:**
   ```bash
   python main_app.py
   ```

6. **Access the system:**
   - Open browser to `http://localhost:5000`
   - Upload videos and start analysis

## Key Features

### 🎥 Video Analysis
- **Object Detection**: YOLO-based detection of people, vehicles, and objects
- **Face Recognition**: FaceNet-powered face detection and encoding
- **License Plate OCR**: EasyOCR for reading license plates from vehicles
- **Frame Extraction**: Intelligent frame sampling for efficient processing

### 🔍 Search Capabilities
- **License Plate Search**: Find vehicles by full or partial plate numbers
- **Face Matching**: Upload a photo to find matching faces in videos
- **Natural Language Queries**: Basic keyword-based search functionality

### 🚨 Smart Alerts
- **Crowd Detection**: Alerts when more than 5 people detected
- **Traffic Congestion**: Alerts for high vehicle density
- **Configurable Thresholds**: Customizable alert conditions

### 📊 Data Management
- **SQLite Database**: Lightweight, local database storage
- **Metadata Tracking**: Frame-by-frame analysis results
- **Performance Metrics**: Processing statistics and confidence scores

## Usage Guidelines

### Video Upload
- Supported formats: MP4, AVI, MOV, etc.
- Maximum file size: 500MB
- Processing time depends on video length and resolution

### Face Matching
- Best results with clear, front-facing photos
- Supports JPG, PNG image formats
- Privacy-focused: images processed locally

### License Plate Search
- Works with partial matches
- Case-insensitive search
- Handles common OCR errors

### System Alerts
- Automatically generated during analysis
- Real-time monitoring capabilities
- Configurable alert thresholds

## Technical Notes

- **AI Models**: Auto-downloads required models on first run
- **Performance**: Optimized for CPU processing (GPU optional)
- **Storage**: All data stored locally for privacy
- **Scalability**: Suitable for small to medium video datasets

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install all required Python packages
2. **Model Download**: First run may take time to download AI models
3. **Video Format**: Ensure video is in supported format
4. **Memory Usage**: Large videos may require significant RAM

### Performance Tips
- Use smaller video files for faster processing
- Lower frame extraction rate for large videos
- Process videos during off-peak hours
- Monitor system resources during analysis