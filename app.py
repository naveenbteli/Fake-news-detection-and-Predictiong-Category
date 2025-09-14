import pickle
from flask import Flask, request, jsonify
import numpy as np
from preprocess import full_preprocess, preprocess_text, category_map

app = Flask(__name__)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('voting_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('category_model.pkl', 'rb') as f:
    category_model = pickle.load(f)

@app.route('/predict', methods=['POST'])

def predict():
    try:
        json_data = request.get_json()

        text = json_data['text']

        article = full_preprocess(text)

        text_transform = vectorizer.transform([article])

        prediction = model.predict(text_transform)

        result = int(prediction)

        label = "false" if result == 1 else "true"
        # true = Fake Article, false = Real Article

        return jsonify({'prediction': label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/category', methods=['POST'])

def predict_category():
    try:

        json_data = request.get_json()

        text = json_data['text']

        # clean_text = preprocess_text(text)

        # text_transform = vectorizer.transform([text])

        # category_pred = category_model.predict(text_transform)

        # category = category_pred[0] 

        # result = str(category)
        
        # return jsonify({'category': result})

        clean_text = preprocess_text(text)

        text_transform = vectorizer.transform([clean_text])

        prediction = category_model.predict(text_transform)[0]

        category = category_map.get(prediction, "UNKNOWN")

        return jsonify({"predicted_category": category})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)