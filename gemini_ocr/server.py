from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from main import process_pdf
import tempfile
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, filename)
        file.save(pdf_path)

        # Process the PDF
        output_path = os.path.join(temp_dir, 'extracted_answers.json')
        process_pdf(pdf_path, output_path)

        # Read the results
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Clean up temporary files
        os.remove(pdf_path)
        os.remove(output_path)
        os.rmdir(temp_dir)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 