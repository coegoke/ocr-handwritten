from flask import Flask, request, jsonify
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import numpy as np
import os

app = Flask(__name__)

# Inisialisasi model OCR
model = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_vgg16_bn', pretrained=True)
model.det_predictor.model.postprocessor.bin_thresh = 0.1
model.det_predictor.model.postprocessor.box_thresh = 0.2

# Fungsi untuk mengambil semua kata dalam struktur data
def extract_words(result):
    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    words.append(word.value)
    return words

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Baca file gambar
        image = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Simpan sementara gambar
        temp_image_path = 'temp_image.jpg'
        cv2.imwrite(temp_image_path, image)

        # Proses OCR
        doc = DocumentFile.from_images(temp_image_path)
        result = model(doc)

        # Hapus file sementara
        os.remove(temp_image_path)

        # Ekstraksi kata-kata dari hasil OCR
        words = extract_words(result)

        return jsonify({"words": words})

if __name__ == '__main__':
    app.run(debug=True)
