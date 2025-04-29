from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import base64
import os

app = Flask(__name__)

# Constants
EYE_AR_THRESH = 0.25  # Threshold for eye aspect ratio to indicate blink
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm drowsiness
YAWN_THRESH = 20  # Threshold for mouth aspect ratio to detect yawns

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    print("Error loading predictor file")
    predictor = None

# Facial landmarks indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    # Compute euclidean distances between vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Compute euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])
    # Compute eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute distances between mouth landmarks
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'status': 'error', 'message': 'Predictor not loaded', 'drowsy': False})
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image provided', 'drowsy': False})
        
        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image', 'drowsy': False})
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        if len(faces) == 0:
            return jsonify({'status': 'success', 'message': 'No face detected', 'drowsy': False})
        
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye and mouth coordinates
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            
            # Calculate eye and mouth aspect ratios
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Check for drowsiness
            if ear < EYE_AR_THRESH or mar > YAWN_THRESH:
                return jsonify({
                    'status': 'success',
                    'message': 'Drowsy detected',
                    'drowsy': True,
                    'ear': float(ear),
                    'mar': float(mar)
                })
        
        return jsonify({
            'status': 'success',
            'message': 'Alert',
            'drowsy': False,
            'ear': float(ear),
            'mar': float(mar)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'drowsy': False})

if __name__ == '__main__':
    app.run(debug=True)