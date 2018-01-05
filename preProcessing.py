import csv
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
# using binary relevance
#from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
seperate_genre=['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western']
def readData(name):
    data = pd.read_csv(name)
    return data


def extract_features(text):
    features = []
    #length of the text
    features.append(len(text))

    features.append(2)
    return features

def getLabels(genres):
    genresN = []
    for g in genres:
        split = [x.strip() for x in g.split(',')]
        l = []
        for s in split:
            index = seperate_genre.index(s)
            l.append(index)
        genresN.append(l)
    labels = MultiLabelBinarizer().fit_transform(genresN)
    return  labels

def toNGrams(text, N):
    # Convert a text to a list of character N-grams.
    array = []
    text = text.lower()
    for i in range(len(text) - (N-1)):
        NGram = ""
        for x in range(N):
            NGram = NGram + text[i+x]
        array.append(NGram)
    return array


def calculateNgrams(train_data):
    global most_bigrams, most_trigrams
    bigrams = []
    trigrams = []
    for train in train_data.data:
        bigrams.append(toNGrams(train, 2))
    for train in train_data.data:
        trigrams.append(toNGrams(train, 3))
    #Create list of most bigrams
    flatten_digram = [item for sublist in bigrams for item in sublist]
    count_bigram = collections.Counter(flatten_digram)
    most_bigrams = count_bigram.most_common(45)
    most_bigrams = [x[0] for x in most_bigrams]
    #Create list of most trigrams
    flatten_trigram = [item for sublist in trigrams for item in sublist]
    count_trigram = collections.Counter(flatten_trigram)
    most_trigrams = count_trigram.most_common(45)
    most_trigrams = [x[0] for x in most_trigrams]



def main():
    data = readData("IMDB-Movie-Data.csv")
    genres = data["Genre"]
    descriptions = data["Description"]
    labels = getLabels(genres)
    features = list(map(extract_features, descriptions))
    # X = features
    # Y = Labels
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print acc

    ''''
    classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
    # train
    classifier.fit(features, labels)
    predictions = classifier.predict(features)
    for p in predictions:
        print p
    '''

if __name__ == '__main__':
    main()

