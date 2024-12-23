import cv2
import numpy as np
import imutils
import requests
from flask import Flask, render_template, Response
from ultralytics import YOLO
import os
app = Flask(__name__)

count_model = YOLO('Models/sku110k_15_epochs.pt')
loreal_model = YOLO('Models/Loreal_50epochs.pt')
url = "http://192.168.15.222:8080/shot.jpg"

def predict_inventory(frame):
    prediction = count_model.predict(frame, project='Temp', name='Photos',conf=0.25)
    return prediction

def predict_loreal(frame):
    prediction = loreal_model.predict(frame, project='Temp', name='Photos', save=False, conf=0.5)
    return prediction
def count(results):
    for result in results:
        dabba = result.boxes
    return len(dabba)

def camera_stream():
    i = 1
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)

        result = predict_inventory(img)
        output = count(result)
        print(output)
        result2 = predict_loreal(img)
        output_folder = f'Temp/Photos{i}/'
        predicted_image_path = os.path.join(output_folder, "image0.jpg")
        if os.path.exists(predicted_image_path):
            img2 = cv2.imread(predicted_image_path)
            cv2.putText(img2, f"  Total Count:{output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', img2)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            i+=1
        else:
            cv2.putText(img, f"  Total Count:{output}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            i+=1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5000)