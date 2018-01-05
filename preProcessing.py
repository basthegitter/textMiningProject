import csv
import numpy as np
import nltk
from nltk.data import load
import sklearn.metrics
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import word_tokenize
import collections
from skmultilearn.adapt import MLkNN
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
#nltk.download('averaged_perceptron_tagger')
tagdict = load('help/tagsets/upenn_tagset.pickle')
most_bigrams = []
most_trigrams = []

seperate_genre=['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western']
def readData(name):
    data = pd.read_csv(name)
    return data


def extract_features(text):
    global most_bigrams, most_trigrams
    pos_tag = nltk.pos_tag(word_tokenize(text))
    features = []
    #length of the text
    features.append(len(text))
    ##POS tags
    # 45 different pos tags
    # Feature 45 - 90: Frequencies of different pos_tags are added
    pos_list = [x[1] for x in pos_tag]
    for tag in tagdict.keys():
        features.append(pos_list.count(tag))

    # 90-135 most common bigrams from the training set
    bigrams = toNGrams(text, 2)

    for bi in most_bigrams:
        features.append(bigrams.count(bi))

    # 135-180 most common trigrams from training set
    trigrams = toNGrams(text, 3)
    for tri in most_trigrams:
        features.append(trigrams.count(tri))
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
    labels = np.array(MultiLabelBinarizer().fit_transform(genresN))
    return labels

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
    for train in train_data:
        bigrams.append(toNGrams(train, 2))
    for train in train_data:
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



def binRel(X_train, X_test, y_test, y_train):
    # initialize binary relevance multi-label classifier
    # with a gaussian naive bayes base classifier
    classifier = BinaryRelevance(GaussianNB())
    # train
    classifier.fit(X_train, y_train)
    # predict
    predictions = classifier.predict(X_test)
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_test, predictions)))

def main():
    data = readData("IMDB-Movie-Data.csv")
    genres = data["Genre"]
    descriptions = data["Description"]
    labels = getLabels(genres)
    calculateNgrams(descriptions)

    features = list(map(extract_features, descriptions))
    print len(features[1])
    # X = features
    # Y = Labels
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    #binRel(X_train, X_test, y_test, y_train)
    classifier = MLkNN(k=4)
    # Train
    classifier.fit(X_train, y_train)
    #predict
    #print X_test
    predictions = classifier.predict(np.array(X_test))
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_test, predictions)))#(y_true, y_pred)

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

