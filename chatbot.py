from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-Neo 2.7B model and tokenizer
model = GPT2LMHeadModel.from_pretrained("DB13067/Peterbot")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Tokenize user input and generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    chatbot_response_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    chatbot_response = tokenizer.decode(chatbot_response_ids[0], skip_special_tokens=True)

    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
