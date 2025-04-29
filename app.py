from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import base64
import os
import time

app = Flask(__name__)

# Constants
EYE_AR_THRESH = 0.25  # Lowered threshold for better sensitivity
EYE_AR_CONSEC_FRAMES = 3  # Reduced consecutive frames for faster response
YAWN_THRESH = 20  # Threshold for mouth aspect ratio to detect yawns
COUNTER = 0
ALARM_ON = False

# Load models
try:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat file.")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Facial landmarks indexes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
except Exception as e:
    print(f"Model Load Error: {e}")
    detector, predictor = None, None

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the horizontal
    # mouth landmarks (x, y)-coordinates
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    
    # Compute the euclidean distance between the vertical
    # mouth landmarks (x, y)-coordinates
    D = distance.euclidean(mouth[12], mouth[16])
    
    # Compute mouth aspect ratio
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
    global COUNTER, ALARM_ON

    if detector is None or predictor is None:
        return jsonify({'label': 'Error: Model not loaded', 'alarm': False})

    try:
        data = request.get_json()
        img_data = data['image']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        npimg = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        label = "Alert"
        alarm = False
        
        if len(subjects) == 0:
            label = "No face detected"
            COUNTER = 0
            ALARM_ON = False
        else:
            for subject in subjects:
                shape = predictor(gray, subject)
                shape = face_utils.shape_to_np(shape)
                
                # Extract eye and mouth coordinates
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                
                # Calculate eye and mouth aspect ratios
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mar = mouth_aspect_ratio(mouth)
                
                # Check for drowsiness (eyes closed)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        label = "Drowsy (Eyes Closed)"
                        alarm = True
                        ALARM_ON = True
                # Check for yawning
                elif mar > YAWN_THRESH:
                    label = "Drowsy (Yawning)"
                    alarm = True
                    ALARM_ON = True
                else:
                    COUNTER = 0
                    ALARM_ON = False
                    label = "Alert"
        
        return jsonify({'label': label, 'alarm': alarm})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'label': 'Error during prediction', 'alarm': False})

if __name__ == '__main__':
    app.run(debug=True)