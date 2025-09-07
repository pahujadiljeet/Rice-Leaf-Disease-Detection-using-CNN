from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)
model = load_model("rice_cnn_model.h5")
classes = ['Leaf_Smut', 'Brown_Spot', 'Bacterial_Leaf_Blight']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = cv2.imread(filepath)
        img_resized = cv2.resize(img, (128,128))
        img_array = np.expand_dims(img_resized/255.0, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = classes[class_index]

        return render_template('index.html', uploaded_image=filepath, prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
