# AI-Video-Analysis-Agent

This project is a Flask-based web application that utilizes computer vision techniques for face recognition and license plate detection using pretrained models (dlib and YOLOv8).

## 🚀 Features

- **Face Recognition** using `dlib_face_recognition_resnet_model_v1`
- **License Plate Detection** using `YOLOv8`
- User-friendly web interface (HTML templates with Flask)
- Search and match functionality for both faces and license plates
- Alert and result display system

## 🗂️ Folder Structure

```
project/
├── main_app.py                  # Main Flask application
├── requirements_txt.txt        # Python dependencies
├── models/
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   └── yolov8n.pt
├── templates/
│   ├── index.html
│   ├── face_match.html
│   ├── plate_search.html
│   ├── results.html
│   └── alerts.html
└── folder_structure.md
```

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-plate-recognition.git
   cd face-plate-recognition/project
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements_txt.txt
   ```

3. Run the app:
   ```bash
   python main_app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/`

## 📚 Requirements

- Python 3.7+
- Flask
- OpenCV
- dlib
- Ultralytics (for YOLOv8)
- Additional packages listed in `requirements_txt.txt`

## 📸 Models Used

- **dlib_face_recognition_resnet_model_v1.dat** – for high-accuracy face recognition
- **yolov8n.pt** – lightweight YOLOv8 model for fast object detection

## 🛠️ Future Improvements

- Add database integration for storing recognized faces/plates
- Real-time video stream support
- Authentication and role-based access

## 📝 License

This project is open-source and available under the MIT License.
