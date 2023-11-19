from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the conversational pipeline from transformers
chatbot = pipeline("conversational")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Use the chatbot for conversational responses
    chatbot_response = chatbot(user_input)[0]["generated_responses"][0]

    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
