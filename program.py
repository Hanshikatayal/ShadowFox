import collections
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. THE BRAIN (The Logic) ---
class AutocorrectSystem:
    def __init__(self):
        self.model = collections.defaultdict(lambda: collections.defaultdict(int))

    def train(self, corpus):
        words = corpus.lower().split()
        for i in range(len(words) - 2):
            context = (words[i], words[i+1])
            next_word = words[i+2]
            self.model[context][next_word] += 1

    def predict(self, text, num_suggestions=3):
        words = text.lower().split()
        if len(words) < 2:
            return []
        
        context = (words[-2], words[-1])
        if context in self.model:
            predictions = self.model[context]
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_predictions[:num_suggestions]]
        return []

# --- 2. THE DATA ---
sample_data = """
    How are you doing today? I am doing great. How are things going? 
    Artificial Intelligence is amazing. Artificial Intelligence is the future.
    The web is a vast place. The web is built with code.
"""

# --- 3. THE SERVER (The Flask App) ---
app = Flask(__name__)
CORS(app)

kb = AutocorrectSystem()
kb.train(sample_data)

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    data = request.json
    user_text = data.get("text", "")
    suggestions = kb.predict(user_text)
    return jsonify({"suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True, port=5000)