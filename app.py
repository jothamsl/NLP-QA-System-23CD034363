from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from LLM_QA_CLI import get_llm_response, preprocess_input

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Ensure API Key is available for the web app
# In a real deployment, this would be set in the environment variables of the hosting provider
if not os.getenv("LLM_API_KEY"):
    print("WARNING: LLM_API_KEY environment variable not set. API calls may fail.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('question')

    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    # 1. Preprocess
    processed_text = preprocess_input(user_input)

    # 2. Get Response
    # We need to handle the case where the CLI script might rely on a global variable set in main()
    # But in our CLI script, get_llm_response checks os.getenv or the global.
    # Since we are in a new process/context, we rely on os.getenv("LLM_API_KEY") being set
    # or we can set it here if we had a config file.

    answer = get_llm_response(processed_text)

    return jsonify({
        'original_question': user_input,
        'processed_question': processed_text,
        'answer': answer
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
