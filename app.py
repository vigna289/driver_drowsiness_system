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
THRESH = 0.25
FRAME_CHECK = 20
flag = 0

# Load model
try:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        raise FileNotFoundError("Missing shape_predictor_68_face_landmarks.dat file.")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
except Exception as e:
    print(f"Model Load Error: {e}")
    detector, predictor = None, None

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    global flag

    if detector is None or predictor is None:
        return jsonify({'label': 'Error: Model not loaded'})

    try:
        data = request.get_json()
        img_data = data['image']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        npimg = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        label = "Alert"
        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < THRESH:
                flag += 1
                if flag >= FRAME_CHECK:
                    label = "Drowsy"
            else:
                flag = 0
                label = "Alert"
        return jsonify({'label': label})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'label': 'Error during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
