from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(app_dir, 'models', 'classifier_model.joblib')
model = joblib.load(model_path)

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img_array = np.array(img).flatten()
    return img_array

@app.route('/')
def index():
    return render_template('index.html', uploaded_images=[])

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html', uploaded_images=[])

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img_array = process_image(file_path)
        prediction = model.predict([img_array])[0]

        valid_categories = ['odol', 'sabun', 'shampo']
        if prediction.lower() in valid_categories:
            return jsonify({'result': prediction, 'result_image': file.filename})
        else:
            return jsonify({'error': 'Bukan termasuk kategori sampah yang valid'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
