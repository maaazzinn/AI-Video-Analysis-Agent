import os
import cv2
import sqlite3
import numpy as np
import json
import base64
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import threading
import time
from pathlib import Path

# Computer Vision and AI imports
try:
    from ultralytics import YOLO
    import easyocr
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install ultralytics easyocr facenet-pytorch torch deep-sort-realtime")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAMES_FOLDER'] = 'static/frames'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/frames', exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('models', exist_ok=True)

class VideoAnalysisAgent:
    def __init__(self):
        self.db_path = 'database/metadata.db'
        self.init_database()
        self.load_models()
        self.alerts = []
        self.face_encodings = {}
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT,
                frame_number INTEGER,
                timestamp REAL,
                objects TEXT,
                face_encodings TEXT,
                license_plates TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                description TEXT,
                frame_number INTEGER,
                video_name TEXT,
                timestamp REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face database table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                encoding TEXT,
                image_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_models(self):
        """Load all AI models"""
        try:
            # YOLO for object detection
            if os.path.exists('models/yolov8n.pt'):
                self.yolo_model = YOLO('models/yolov8n.pt')
            else:
                print("Downloading YOLOv8 model...")
                self.yolo_model = YOLO('yolov8n.pt')  # Will auto-download
                
            # FaceNet for face recognition
            self.mtcnn = MTCNN(keep_all=True)
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            
            # EasyOCR for license plate reading
            self.ocr_reader = easyocr.Reader(['en'])
            
            # DeepSORT for tracking
            self.tracker = DeepSort(max_age=50, n_init=3)
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.yolo_model = None
            self.mtcnn = None
            self.facenet = None
            self.ocr_reader = None
            self.tracker = None
    
    def extract_frames(self, video_path, fps=1):
        """Extract frames from video at specified FPS"""
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = f"{app.config['FRAMES_FOLDER']}/frame_{frame_count}.jpg"
                cv2.imwrite(frame_path, frame)
                frames.append({
                    'frame_number': frame_count,
                    'timestamp': frame_count / video_fps,
                    'path': frame_path,
                    'frame': frame
                })
                
            frame_count += 1
            
        cap.release()
        return frames
    
    def get_frame_image_path(self, video_name, frame_number):
        """Get the path to a specific frame image"""
        # Construct frame path based on naming convention
        frame_filename = f"frame_{frame_number}.jpg"
        frame_path = os.path.join(app.config['FRAMES_FOLDER'], frame_filename)
        
        # If the original frame doesn't exist, try to extract it from video
        if not os.path.exists(frame_path):
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
            if os.path.exists(video_path):
                self.extract_specific_frame(video_path, frame_number, frame_path)
        
        return frame_path if os.path.exists(frame_path) else None
    
    def extract_specific_frame(self, video_path, frame_number, output_path):
        """Extract a specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(output_path, frame)
        cap.release()
    
    def draw_face_boxes_on_frame(self, frame_path, face_bboxes, output_path=None):
        """Draw bounding boxes around detected faces on frame image"""
        if not os.path.exists(frame_path):
            return None
            
        image = cv2.imread(frame_path)
        if image is None:
            return None
        
        # Draw rectangles around faces
        for bbox in face_bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, 'Match', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            return output_path
        else:
            # Return base64 encoded image for direct display
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
    
    def get_face_crop_from_frame(self, video_name, frame_number, face_bbox):
        """Extract just the face region from a frame"""
        frame_path = self.get_frame_image_path(video_name, frame_number)
        if not frame_path or not os.path.exists(frame_path):
            return None
        
        image = cv2.imread(frame_path)
        if image is None:
            return None
        
        x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
        
        # Add some padding around the face
        padding = 20
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        face_crop = image[y1:y2, x1:x2]
        return face_crop
    
    def analyze_frame(self, frame, frame_info):
        """Analyze single frame for objects, faces, and license plates"""
        results = {
            'objects': [],
            'faces': [],
            'license_plates': [],
            'face_encodings': []
        }
        
        if self.yolo_model is None:
            return results
            
        # YOLO object detection
        detections = self.yolo_model(frame)
        
        for detection in detections:
            boxes = detection.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.yolo_model.names[cls]
                    
                    if conf > 0.5:  # Confidence threshold
                        obj_info = {
                            'label': label,
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        results['objects'].append(obj_info)
                        
                        # Process vehicles for license plates
                        if label in ['car', 'truck', 'bus', 'motorcycle']:
                            self.process_vehicle(frame, obj_info, results)
                            
                        # Process persons for face detection
                        if label == 'person':
                            self.process_person(frame, obj_info, results)
        
        return results
    
    def process_vehicle(self, frame, vehicle_info, results):
        """Extract and read license plates from vehicles"""
        if self.ocr_reader is None:
            return
            
        x1, y1, x2, y2 = vehicle_info['bbox']
        vehicle_crop = frame[y1:y2, x1:x2]
        
        try:
            # OCR on vehicle crop
            ocr_results = self.ocr_reader.readtext(vehicle_crop)
            
            for (bbox, text, conf) in ocr_results:
                if conf > 0.7 and len(text) > 4:  # Filter for likely license plates
                    plate_info = {
                        'text': text.upper().replace(' ', ''),
                        'confidence': conf,
                        'vehicle_bbox': vehicle_info['bbox']
                    }
                    results['license_plates'].append(plate_info)
                    
        except Exception as e:
            print(f"OCR Error: {e}")
    
    def process_person(self, frame, person_info, results):
        """Detect and encode faces from person detections"""
        if self.mtcnn is None or self.facenet is None:
            return
            
        x1, y1, x2, y2 = person_info['bbox']
        person_crop = frame[y1:y2, x1:x2]
        
        try:
            # Detect faces in person crop
            faces, _ = self.mtcnn.detect(person_crop)
            
            if faces is not None:
                for face_bbox in faces:
                    if face_bbox is not None:
                        fx1, fy1, fx2, fy2 = [int(coord) for coord in face_bbox]
                        face_crop = person_crop[fy1:fy2, fx1:fx2]
                        
                        # Encode face
                        face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float().unsqueeze(0)
                        face_tensor = torch.nn.functional.interpolate(face_tensor, size=(160, 160))
                        face_tensor = (face_tensor - 127.5) / 128.0
                        
                        with torch.no_grad():
                            encoding = self.facenet(face_tensor).cpu().numpy()[0]
                            
                        face_info = {
                            'bbox': [fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1],
                            'encoding': encoding.tolist()
                        }
                        results['faces'].append(face_info)
                        results['face_encodings'].append(encoding.tolist())
                        
        except Exception as e:
            print(f"Face processing error: {e}")
    
    def save_frame_metadata(self, video_name, frame_info, analysis_results):
        """Save frame analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO frame_metadata 
            (video_name, frame_number, timestamp, objects, face_encodings, license_plates)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            video_name,
            frame_info['frame_number'],
            frame_info['timestamp'],
            json.dumps(analysis_results['objects']),
            json.dumps(analysis_results['face_encodings']),
            json.dumps(analysis_results['license_plates'])
        ))
        
        conn.commit()
        conn.close()
        
        # Check for alerts
        self.check_alerts(video_name, frame_info, analysis_results)
    
    def check_alerts(self, video_name, frame_info, analysis_results):
        """Check for alert conditions and trigger alerts"""
        # Count people in frame
        person_count = len([obj for obj in analysis_results['objects'] if obj['label'] == 'person'])
        
        if person_count > 5:
            self.trigger_alert(
                'crowd_threshold',
                f'Crowd threshold exceeded: {person_count} people detected',
                frame_info['frame_number'],
                video_name,
                frame_info['timestamp']
            )
        
        # Check for specific vehicles (you can customize this)
        vehicles = [obj for obj in analysis_results['objects'] if obj['label'] in ['car', 'truck', 'bus']]
        if len(vehicles) > 3:
            self.trigger_alert(
                'vehicle_congestion',
                f'High vehicle density: {len(vehicles)} vehicles detected',
                frame_info['frame_number'],
                video_name,
                frame_info['timestamp']
            )
    
    def trigger_alert(self, alert_type, description, frame_number, video_name, timestamp):
        """Trigger and save alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, description, frame_number, video_name, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (alert_type, description, frame_number, video_name, timestamp))
        
        conn.commit()
        conn.close()
        
        alert_info = {
            'type': alert_type,
            'description': description,
            'frame_number': frame_number,
            'video_name': video_name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat()
        }
        self.alerts.append(alert_info)
        print(f"ALERT: {description}")
    
    def process_video(self, video_path, video_name):
        """Main video processing pipeline"""
        print(f"Starting analysis of {video_name}")
        
        # Extract frames
        frames = self.extract_frames(video_path, fps=1)
        
        for i, frame_info in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            
            # Analyze frame
            analysis_results = self.analyze_frame(frame_info['frame'], frame_info)
            
            # Save results
            self.save_frame_metadata(video_name, frame_info, analysis_results)
        
        print(f"Analysis complete for {video_name}")
        return len(frames)
    
    def search_license_plate(self, plate_query):
        """Search for license plate in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_name, frame_number, timestamp, license_plates
            FROM frame_metadata
            WHERE license_plates LIKE ?
        ''', (f'%{plate_query.upper()}%',))
        
        results = []
        for row in cursor.fetchall():
            video_name, frame_number, timestamp, plates_json = row
            plates = json.loads(plates_json)
            
            for plate in plates:
                if plate_query.upper() in plate['text']:
                    results.append({
                        'video_name': video_name,
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'plate_text': plate['text'],
                        'confidence': plate['confidence']
                    })
        
        conn.close()
        return results
    
    def match_face(self, face_encoding, threshold=0.6):
        """Match face encoding against database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT video_name, frame_number, timestamp, face_encodings FROM frame_metadata')
        
        matches = []
        for row in cursor.fetchall():
            video_name, frame_number, timestamp, encodings_json = row
            encodings = json.loads(encodings_json)
            
            for encoding in encodings:
                # Calculate cosine similarity
                similarity = np.dot(face_encoding, encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(encoding)
                )
                
                if similarity > threshold:
                    matches.append({
                        'video_name': video_name,
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'similarity': similarity
                    })
        
        conn.close()
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def match_face_with_images(self, face_encoding, threshold=0.6):
        """Match face encoding against database and include frame image paths"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT video_name, frame_number, timestamp, face_encodings FROM frame_metadata')
        
        matches = []
        for row in cursor.fetchall():
            video_name, frame_number, timestamp, encodings_json = row
            encodings = json.loads(encodings_json)
            
            for encoding in encodings:
                # Calculate cosine similarity
                similarity = np.dot(face_encoding, encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(encoding)
                )
                
                if similarity > threshold:
                    # Get frame image path
                    frame_image_path = self.get_frame_image_path(video_name, frame_number)
                    frame_image_url = None
                    
                    if frame_image_path and os.path.exists(frame_image_path):
                        # Convert absolute path to relative URL path
                        frame_image_url = frame_image_path.replace('static/', '')
                    
                    matches.append({
                        'video_name': video_name,
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'similarity': similarity,
                        'frame_image_url': frame_image_url
                    })
        
        conn.close()
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def match_face_with_detailed_info(self, face_encoding, threshold=0.6):
        """Enhanced face matching with additional metadata and face locations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT video_name, frame_number, timestamp, face_encodings, objects 
            FROM frame_metadata 
            WHERE face_encodings != "[]"
        ''')
        
        matches = []
        for row in cursor.fetchall():
            video_name, frame_number, timestamp, encodings_json, objects_json = row
            encodings = json.loads(encodings_json)
            objects = json.loads(objects_json)
            
            # Count people in frame for context
            person_count = len([obj for obj in objects if obj['label'] == 'person'])
            
            for i, encoding in enumerate(encodings):
                # Calculate cosine similarity
                similarity = np.dot(face_encoding, encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(encoding)
                )
                
                if similarity > threshold:
                    # Get frame image path
                    frame_image_path = self.get_frame_image_path(video_name, frame_number)
                    frame_image_url = None
                    
                    if frame_image_path and os.path.exists(frame_image_path):
                        frame_image_url = frame_image_path.replace('static/', '')
                    
                    matches.append({
                        'video_name': video_name,
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'similarity': similarity,
                        'frame_image_url': frame_image_url,
                        'face_index': i,
                        'people_in_frame': person_count,
                        'confidence_level': 'High' if similarity > 0.8 else 'Medium' if similarity > 0.6 else 'Low'
                    })
        
        conn.close()
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

# Initialize the agent
agent = VideoAnalysisAgent()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No video file selected')
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        flash('No video file selected')
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start video processing in background
        def process_in_background():
            agent.process_video(filepath, filename)
        
        thread = threading.Thread(target=process_in_background)
        thread.start()
        
        flash(f'Video {filename} uploaded and processing started!')
        return redirect(url_for('results'))

@app.route('/results')
def results():
    conn = sqlite3.connect(agent.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT video_name, COUNT(*) as frame_count, 
               MAX(timestamp) as duration
        FROM frame_metadata 
        GROUP BY video_name
        ORDER BY created_at DESC
    ''')
    
    videos = []
    for row in cursor.fetchall():
        videos.append({
            'name': row[0],
            'frame_count': row[1],
            'duration': row[2]
        })
    
    conn.close()
    return render_template('results.html', videos=videos)

@app.route('/face_match', methods=['GET', 'POST'])
def face_match():
    if request.method == 'POST':
        if 'face_image' not in request.files:
            flash('No face image uploaded')
            return redirect(url_for('face_match'))
        
        file = request.files['face_image']
        if file.filename == '':
            flash('No face image selected')
            return redirect(url_for('face_match'))
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process uploaded face
        image = cv2.imread(filepath)
        if agent.mtcnn and agent.facenet:
            try:
                faces, _ = agent.mtcnn.detect(image)
                if faces is not None and len(faces) > 0:
                    face_bbox = faces[0]
                    x1, y1, x2, y2 = [int(coord) for coord in face_bbox]
                    face_crop = image[y1:y2, x1:x2]
                    
                    face_tensor = torch.tensor(face_crop).permute(2, 0, 1).float().unsqueeze(0)
                    face_tensor = torch.nn.functional.interpolate(face_tensor, size=(160, 160))
                    face_tensor = (face_tensor - 127.5) / 128.0
                    
                    with torch.no_grad():
                        encoding = agent.facenet(face_tensor).cpu().numpy()[0]
                    
                    # Use the new method that includes frame images
                    matches = agent.match_face_with_images(encoding)
                    return render_template('face_match.html', matches=matches, uploaded_image=filename)
                else:
                    flash('No face detected in uploaded image')
            except Exception as e:
                flash(f'Error processing face: {e}')
        else:
            flash('Face recognition models not loaded')
    
    return render_template('face_match.html')

@app.route('/plate_search', methods=['GET', 'POST'])
def plate_search():
    if request.method == 'POST':
        plate_query = request.form.get('plate_number', '').strip()
        if plate_query:
            matches = agent.search_license_plate(plate_query)
            return render_template('plate_search.html', matches=matches, query=plate_query)
        else:
            flash('Please enter a license plate number')
    
    return render_template('plate_search.html')

@app.route('/alerts')
def alerts():
    conn = sqlite3.connect(agent.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT alert_type, description, frame_number, video_name, timestamp, created_at
        FROM alerts
        ORDER BY created_at DESC
        LIMIT 50
    ''')
    
    alerts = []
    for row in cursor.fetchall():
        alerts.append({
            'type': row[0],
            'description': row[1],
            'frame_number': row[2],
            'video_name': row[3],
            'timestamp': row[4],
            'created_at': row[5]
        })
    
    conn.close()
    return render_template('alerts.html', alerts=alerts)

@app.route('/highlighted_frame/<video_name>/<int:frame_number>')
def highlighted_frame(video_name, frame_number):
    """Serve frame image with face detection boxes highlighted"""
    frame_path = agent.get_frame_image_path(video_name, frame_number)
    if not frame_path or not os.path.exists(frame_path):
        return "Frame not found", 404
    
    # Get face data for this frame
    conn = sqlite3.connect(agent.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT objects FROM frame_metadata 
        WHERE video_name = ? AND frame_number = ?
    ''', (video_name, frame_number))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        objects = json.loads(result[0])
        face_boxes = []
        
        # Extract face bounding boxes from person detections
        for obj in objects:
            if obj['label'] == 'person':
                # This is simplified - you'd need to store actual face boxes
                # For now, we'll use the person bounding box
                face_boxes.append(obj['bbox'])
        
        # Create highlighted version
        highlighted_path = frame_path.replace('.jpg', '_highlighted.jpg')
        agent.draw_face_boxes_on_frame(frame_path, face_boxes, highlighted_path)
        
        if os.path.exists(highlighted_path):
            return send_file(highlighted_path)
    
    # Fall back to original frame
    return send_file(frame_path)

@app.route('/api/query', methods=['POST'])
def natural_language_query():
    """Handle natural language queries (simplified version)"""
    query = request.json.get('query', '')
    
    # Simple keyword-based query processing
    # In a full implementation, you'd use an LLM here
    results = []
    
    if 'person' in query.lower() or 'people' in query.lower():
        conn = sqlite3.connect(agent.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT video_name, frame_number, timestamp, objects
            FROM frame_metadata
            WHERE objects LIKE '%person%'
            LIMIT 20
        ''')
        
        for row in cursor.fetchall():
            objects = json.loads(row[3])
            person_count = len([obj for obj in objects if obj['label'] == 'person'])
            results.append({
                'video_name': row[0],
                'frame_number': row[1],
                'timestamp': row[2],
                'description': f'{person_count} people detected'
            })
        conn.close()
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)