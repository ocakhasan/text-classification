from flask import Flask, render_template, request, redirect, flash, url_for
import pickle
import os
from nltk.corpus import stopwords
import numpy as np


STOP_WORDS= list(stopwords.words('turkish'))

app = Flask(__name__)

app.secret_key = "secret key"

cwd = os.getcwd()
dataset_path = os.path.join(cwd, "dataset")
models_path = os.path.join(cwd, "models")

tfidf_path = os.path.join(dataset_path,  "tfidf.pickle")
with open(tfidf_path, 'rb') as data:
    tfidf = pickle.load(data)

mnbc_path = os.path.join(models_path,  "mnbc.pickle")
with open(mnbc_path, 'rb') as data:
    mnbc = pickle.load(data)

labels = {
    'Economy': 1,
    'Education': 2,
    'Politics': 3,
    'Relationships': 4,
    'Sports': 5
}

def clean_the_tweet(sentence):
    sentence = sentence.replace("\r", " ")
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("    ", " ")
    sentence = sentence.replace('"', '')
    sentence = sentence.replace('"', '')

    punctuation_signs = list(")(?:!.,;")

    for punct_sign in punctuation_signs:
        sentence = sentence.replace(punct_sign, ' ')

    for stop_word in STOP_WORDS:
        regex_stopword = r"\b" + stop_word + r"\b"
        sentence = sentence.replace(regex_stopword, '')

    print("Data cleaning is done")
    return sentence

def get_predictions(tweets):

    tweet_features = tfidf.transform(tweets)
    labels_keys = list(labels.keys())
    prediction = mnbc.predict(tweet_features)

    return labels_keys[int(prediction) - 1]


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    sentence=""
    text=""
    if request.method == "GET":
        print("Request is GET")

    else:
        req = request.form
        text = req['text'].lower()
        text=clean_the_tweet(text)
        predicted_array = np.array([text])
        result = get_predictions(predicted_array)
        print(text)
        print(result)
        print("Request is POST")

    return render_template("index.html", result=result, sentence=text)
