from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import base64

app = Flask(__name__)

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
YAWN_THRESH = 20

# Initialize counters
frame_counter = 0

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmarks indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    global frame_counter
    
    try:
        # Get image from request
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 0)
        if len(faces) == 0:
            frame_counter = 0
            return jsonify({'drowsy': False})
        
        # Analyze first face found
        shape = predictor(gray, faces[0])
        shape = face_utils.shape_to_np(shape)
        
        # Calculate EAR and MAR
        ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0
        mar = mouth_aspect_ratio(shape[mStart:mEnd])
        
        # Check for drowsiness
        if ear < EYE_AR_THRESH or mar > YAWN_THRESH:
            frame_counter += 1
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                return jsonify({'drowsy': True})
        else:
            frame_counter = 0
            
        return jsonify({'drowsy': False})
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'drowsy': False})

if __name__ == '__main__':
    app.run(debug=True)