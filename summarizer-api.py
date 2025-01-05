from flask import Flask, request, jsonify
from utils import generate_summary_bart, generate_summary_pegasus, extract_text_from_pdf, question_answering
import PyPDF2
import os
import tempfile

app = Flask(__name__)

# Global variable to store the text
global_text = None

@app.route('/summarize/bart', methods=['POST'])
def summarize_bart():
    global global_text
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body."}), 400

        global_text = data['text']
        summary = generate_summary_bart(global_text)

        if summary is None:
            return jsonify({"error": "Failed to generate summary with BART."}), 500

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/summarize/pegasus', methods=['POST'])
def summarize_pegasus():
    global global_text
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body."}), 400

        global_text = data['text']
        summary = generate_summary_pegasus(global_text)

        if summary is None:
            return jsonify({"error": "Failed to generate summary with Pegasus."}), 500

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_from_pdf', methods=['POST'])
def get_from_pdf():
    global global_text
    try:
        pdf_url = request.form.get('pdf_url')
        if pdf_url:
            global_text = extract_text_from_pdf(pdf_url)
        else:
            uploaded_file = request.files.get('pdf_file')
            if uploaded_file and uploaded_file.filename.endswith('.pdf'):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.filename)
                uploaded_file.save(temp_path)
                try:
                    with open(temp_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        global_text = ""
                        for page in range(len(reader.pages)):
                            global_text += reader.pages[page].extract_text()
                    os.remove(temp_path)
                except Exception as e:
                    return jsonify({"error": f"Error processing the file: {str(e)}"}), 500
            else:
                return jsonify({"error": "Please provide a valid PDF file."}), 400

        return jsonify({"text": global_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_from_pdf_file', methods=['POST'])
def get_from_pdf_file():
    global global_text
    try:
        uploaded_file = request.files.get('pdf_file')
        if uploaded_file and uploaded_file.filename.endswith('.pdf'):
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.filename)
            uploaded_file.save(temp_path)

            try:
                with open(temp_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    global_text = ""
                    for page in range(len(reader.pages)):
                        global_text += reader.pages[page].extract_text()
            finally:
                os.remove(temp_path)

            return jsonify({"text": global_text})
        else:
            return jsonify({"error": "Please upload a valid PDF file."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/get_from_pdf_url', methods=['POST'])
def get_from_pdf_url():
    global global_text
    try:
        pdf_url = request.json.get('pdf_url')
        if pdf_url:
            global_text = extract_text_from_pdf(pdf_url)
            return jsonify({"text": global_text})
        else:
            return jsonify({"error": "No 'pdf_url' provided in request body."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/question_answering', methods=['POST'])
def answer():
    global global_text
    try:
        if not global_text:
            return jsonify({"error": "No text available. Please summarize or extract text first."}), 400

        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body."}), 400

        question = data['question']
        answer = question_answering(global_text, question)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
