import pickle
import numpy as np
import os

labels = {
    'Economy': 1,
    'Education': 2,
    'Politics': 3,
    'Relationships': 4,
    'Sports': 5
}

cwd = os.getcwd()
dataset_path = os.path.join(cwd, "dataset")
models_path = os.path.join(cwd, "models")

path_tfidf = os.path.join(dataset_path,  "tfidf.pickle")
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

path_model = os.path.join(models_path,  "mnbc.pickle")
with open(path_model, 'rb') as data:
    mnbc = pickle.load(data)

def get_predictions(tweets):

    tweet_features = tfidf.transform(tweets)
    labels_keys = list(labels.keys())
    prediction = mnbc.predict(tweet_features)

    for index, tweet in enumerate(my_tweets):
        category = labels_keys[prediction[index] - 1]
        cur_tweet = my_tweets[index]

        print("{} ----->> {}".format(cur_tweet, category))


if __name__ == "__main__":
    my_tweets = np.array(["Doların bu kadar yükselmesi bizim için kötü oldu. Her geçen gün bilgisayar fiyatları artıyor.",
                      "Sınavın kolay olması üniversite sıralamalarını çok değiştirdi. Öğrenciler ne yapacaklarını şaşırdı.",
                      "Derbinin kazananı Beşiktaş oldu ve ligi 3.sırada bitirdi."])

    get_predictions(my_tweets)