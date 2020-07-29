import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
from sklearn.feature_selection import chi2



#Folder for our data
dataset_path = os.path.join(os.getcwd(), "dataset")
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

STOP_WORDS = list(stopwords.words('turkish'))

labels = {
    'Economy': 1,
    'Education': 2,
    'Politics': 3,
    'Relationships': 4,
    'Sports': 5
}

def show_some_keywords(tfidf, features_train, labels_train ):
    for label, category_id in sorted(labels.items()):
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}' category:".format(label))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("")


def clean_dataset(df):

    df['text'] = df['text'].str.replace("\r", " ")
    df['text'] = df['text'].str.replace("\n", " ")
    df['text'] = df['text'].str.replace("    ", " ")
    df['text'] = df['text'].str.replace('"', '')
    df['text'] = df['text'].str.replace('"', '')

    punctuation_signs = list(")(?:!.,;")

    for punct_sign in punctuation_signs:
        df['text'] = df['text'].str.replace(punct_sign, ' ')

    for stop_word in STOP_WORDS:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['text'] = df['text'].str.replace(regex_stopword, '')

    print("Data cleaning is done")
    return df

# Use tfidf to create numeric dataset from the text.


def create_dataset(X_train, X_test, y_train, y_test):
    # Parameters for TFIDF Vectorizer
    ngram_range = (1, 2)
    min_df = 10
    max_df = 1.
    max_features = 1000

    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features,
                            norm='l2',
                            sublinear_tf=True)

    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    print("Labels train shape is ", labels_train.shape)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test

    show_some_keywords(tfidf, features_train, labels_train)

    # features_train
    with open('dataset/features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)

    # labels_train
    with open('dataset/labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)

    # features_test
    with open('dataset/features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)

    # labels_test
    with open('dataset/labels_test.pickle', 'wb') as output:
        pickle.dump(labels_test, output)
        print("labels_test shape is ", labels_test.shape)
    
    # TF-IDF object
    with open('dataset/tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf, output)
    
    print("TF-idf process is done")


def get_dataset():
    df = pd.read_csv('real.csv')
    df = clean_dataset(df)

    df['label'] = df['label'].map(labels)

    X_train, X_test, y_train, y_test = train_test_split(df['text'],
                                                        df['label'],
                                                        test_size=0.20,
                                                        random_state=8)

    create_dataset(X_train, X_test, y_train, y_test)

    print("Process is done")


if __name__ == "__main__":
    get_dataset()
