from flask import Flask, request, jsonify
from utils import generate_summary_bart, generate_summary_pegasus

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)
