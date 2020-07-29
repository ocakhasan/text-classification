import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import os

cwd = os.getcwd()
models_path = os.path.join(cwd, "models")
if not os.path.exists(models_path):
    os.mkdir(models_path)

dataset_path = os.path.join(cwd, "dataset")

#We will use naive bayes for the classification task


def load_data():
    print("Loading data is starting")
    # features_train
    path_features_train = os.path.join(dataset_path,  "features_train.pickle")
    with open(path_features_train, 'rb') as data:
        features_train = pickle.load(data)

    # labels_train
    path_labels_train = os.path.join(dataset_path,  "labels_train.pickle")
    with open(path_labels_train, 'rb') as data:
        labels_train = pickle.load(data)

    # features_test
    path_features_test = os.path.join(dataset_path,  "features_test.pickle")
    with open(path_features_test, 'rb') as data:
        features_test = pickle.load(data)

    # labels_test
    path_labels_test = os.path.join(dataset_path,  "labels_test.pickle")
    with open(path_labels_test, 'rb') as data:
        labels_test = pickle.load(data)

    print("Loading data is done\n")
    return  features_train, features_test, labels_train, labels_test

def train_model(features_train, labels_train, features_test, labels_test):

    print("Model training is starting\n")
    mnbc = MultinomialNB()

    mnbc.fit(features_train, labels_train)
    mnbc_pred = mnbc.predict(features_test)


    print("The training accuracy is: ")
    print(accuracy_score(labels_train, mnbc.predict(features_train)),"\n")

    # Test accuracy
    print("The test accuracy is: ")
    print(accuracy_score(labels_test, mnbc_pred), "\n")

    # Classification report
    print("Classification report")
    print(classification_report(labels_test,mnbc_pred),"\n")

    print("Confusion matrix")
    print(confusion_matrix(labels_test, mnbc_pred),"\n")

    #Save the model
    with open('models/mnbc.pickle', 'wb') as output:
        pickle.dump(mnbc, output)

    print("Model training is done, model is saved to models folder")

if __name__ == "__main__":
    features_train, features_test, labels_train, labels_test = load_data()
    train_model(features_train, labels_train, features_test, labels_test)