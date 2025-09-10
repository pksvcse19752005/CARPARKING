from flask import Flask, request, send_file
from flask_cors import CORS
import torch
import numpy as np
import cv2
import io
from PIL import Image

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return {"error": "No image sent"}, 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    results = model(img_np)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    cars = []
    for label, cord in zip(labels, cords):
        if model.names[int(label)] == 'car' and cord[4] > 0.3:
            cars.append(cord)

    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    for cord in cars:
        x1, y1, x2, y2 = int(cord[0]*w), int(cord[1]*h), int(cord[2]*w), int(cord[3]*h)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)

    _, buffer = cv2.imencode('.jpg', img_cv)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
