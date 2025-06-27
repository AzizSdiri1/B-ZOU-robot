from flask import Flask, Response
import cv2

app = Flask(__name__)

@app.route('/capture', methods=['GET'])
def capture_image():
    # Open the camera inside the route to get the most current frame
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return "Camera not accessible", 500

    # Warm up the camera: read a few frames to flush the buffer
    for _ in range(5):
        ret, frame = camera.read()
        if not ret:
            camera.release()
            return "Failed to grab frame", 500

    # Use the last frame for response
    ret, jpeg = cv2.imencode('.jpg', frame)
    camera.release()

    if not ret:
        return "Failed to encode frame", 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/')
def home():
    return 'Camera API is running. Use /capture to get an image.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
