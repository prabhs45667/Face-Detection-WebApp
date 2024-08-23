from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, static_folder='../frontend')

# Load models
liveness_model = tf.keras.models.load_model('liveness.model')
#liveness_model = load_model('liveness.h5')
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

def detect_and_predict_liveness(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = tf.expand_dims(face, axis=0)

            preds.append(liveness_model.predict(face)[0])
            faces.append((startX, startY, endX, endY))

    return (faces, preds)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    (faces, preds) = detect_and_predict_liveness(image)

    results = []
    for (box, pred) in zip(faces, preds):
        (startX, startY, endX, endY) = box
        label = "Live" if pred[0] > 0.5 else "Fake"
        results.append({"box": (int(startX), int(startY), int(endX), int(endY)), "label": label})

    return jsonify(results)

# Serve the frontend
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)