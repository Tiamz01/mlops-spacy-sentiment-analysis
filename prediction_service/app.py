import os
import requests
from flask import Flask
from flask import request
from flask import jsonify

import spacy
MODEL_DIR = './model/model-best/model.spacy'
nlp = spacy.load(f"{MODEL_DIR}")

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')

from preprocess import preprocessing
# from predict import SENTIMENT_THRESHOLD, SENTIMENT_THRESHOLD2
from predict import spacy_get_sentiment, spacy_get_sentiment_preprocess, spacy_test_text, spacy_test_list

app = Flask('online-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    object = request.get_json()
    print('Received request:', object)
    
    doc, result = spacy_test_text(nlp, preprocessing(object['text']), verbose=True)

    # send_to_evidently_service(object, result)
    return jsonify(result)

def send_to_evidently_service(object, result):
    obj = object.copy()
    obj['prediction'] = result
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/responce", json=[obj])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
