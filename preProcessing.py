import csv
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
    features.append(1)
    features.append(2)
    return features

def getGenres(data):
    genres = data["Genre"]
    genresN = []
    for g in genres:
        split = [x.strip() for x in g.split(',')]
        l = []
        for s in split:
            index = seperate_genre.index(s)
            l.append(index)
        genresN.append(l)
    return  genresN
'''
def read_data(name):
    with open(name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        descriptions = []
        genres = []
        for row in readCSV:
            genres.append(row[2])
            descriptions.append(row[3])

    return descriptions, genres
'''
data = readData("IMDB-Movie-Data.csv")

genresN = getGenres(data)
descriptions = data["Description"]
print descriptions[1]
labels = MultiLabelBinarizer().fit_transform(genresN)
features = list(map(extract_features, descriptions))
print features
''''
classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
# train
classifier.fit(features, labels)
predictions = classifier.predict(features)
for p in predictions:
    print p
'''
#features = list(map(extract_features, train_data.data))

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())
# train
classifier.fit(features, labels)
# predict
predictions = classifier.predict(features)
acc = accuracy_score(labels, predictions)

print acc