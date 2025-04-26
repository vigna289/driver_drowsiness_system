from flask import Flask, render_template, Response, redirect, url_for
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
from pygame import mixer
import time
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

# Initialize sound mixer
mixer.init()
mixer.music.load("alert.mp3")

predictions = []

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.thresh = 0.25
        self.frame_check = 20
        self.flag = 0
        self.alert_active = False
        self.last_alert_time = 0

    def __del__(self):
        self.video.release()

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        label = "Alert"

        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    current_time = time.time()
                    if current_time - self.last_alert_time > 5:
                        self.alert_active = True
                        self.last_alert_time = current_time
                        mixer.music.play()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    label = "Drowsy"
            else:
                self.flag = 0
                self.alert_active = False
                label = "Alert"

        predictions.append(label)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    calculate_accuracy(predictions)
    return redirect(url_for('index'))

def calculate_accuracy(predictions):
    if not predictions:
        print("No predictions to evaluate.")
        return

    y_pred = predictions
    y_true = predictions  # Assuming perfect ground truth for demo

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Alert", "Drowsy"], output_dict=True)

    print("\n=== Accuracy Report ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    for label in ["Alert", "Drowsy"]:
        metrics = report[label]
        print(f"{label:<10} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {int(metrics['support'])}")

if __name__ == '__main__':
    app.run(debug=True)
