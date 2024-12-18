from flask import Flask, request, jsonify
from utils import generate_summary_bart, generate_summary_pegasus, extract_text_from_pdf
import PyPDF2
import os 
import tempfile
import requests

app = Flask(__name__)

@app.route('/summarize/bart', methods=['POST'])
def summarize_bart():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body."}), 400

        text = data['text']
        summary = generate_summary_bart(text)

        if summary is None:
            return jsonify({"error": "Failed to generate summary with BART."}), 500

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/summarize/pegasus', methods=['POST'])
def summarize_pegasus():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body."}), 400

        text = data['text']
        summary = generate_summary_pegasus(text)

        if summary is None:
            return jsonify({"error": "Failed to generate summary with Pegasus."}), 500

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_from_pdf',methods=['POST'])
def get_from_pdf():
    pdf_url = request.form.get('pdf_url')
    if pdf_url:
        text = extract_text_from_pdf(pdf_url)
    else:
        uploaded_file = request.files.get('pdf_file')
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            # Save the uploaded file to a temporary location
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.filename)
            uploaded_file.save(temp_path)
            try:
                # Extract text from the saved PDF file
                with open(temp_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in range(len(reader.pages)):
                        text += reader.pages[page].extract_text()
                # Once the text has been extracted, delete the temporary file
                os.remove(temp_path)
            except Exception as e:
                # Handle any error that occurs during processing
                return f"Error processing the file: {str(e)}"
        else:
            return "Please provide a valid PDF file."

@app.route('/get_from_pdf_file', methods=['POST'])
def get_from_pdf_file():
    try:
        uploaded_file = request.files.get('pdf_file')
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            # Save the uploaded file to a temporary location
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.filename)
            uploaded_file.save(temp_path)

            try:
                # Extract text from the saved PDF file
                with open(temp_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in range(len(reader.pages)):
                        text += reader.pages[page].extract_text()
            finally:
                # Delete the temporary file
                os.remove(temp_path)

            return jsonify({"text": text})
        else:
            return jsonify({"error": "Please upload a valid PDF file."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
@app.route('/get_from_pdf_url', methods=['POST'])
def get_from_pdf_url():
    try:
        pdf_url = request.json.get('pdf_url')
        if pdf_url:
            text = extract_text_from_pdf(pdf_url)
            return jsonify({"text": text})
        else:
            return jsonify({"error": "No 'pdf_url' provided in request body."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)




