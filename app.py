
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load pre-trained OpenCV model (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()  # Read the frame
        if not success:
            break
        else:
            # Convert to grayscale and detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Encode frame for HTML streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Frame format

@app.route('/')
def index():
    return render_template('index.html')  # Load HTML page

@app.route('/video_feed')
def video_feed():
    # Route for video streaming
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
