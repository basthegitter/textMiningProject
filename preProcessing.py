import csv
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

seperate_genre=['Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western']

def readData(name):
    data = pd.read_csv(name)
    return data

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
binary = MultiLabelBinarizer().fit_transform(genresN)
features = []

X = features.data
Y = genresN.target

print binary[0:5]











#d, g = read_data('IMDB-Movie-Data.csv')
#str = g[1]
#print str
#[x.strip() for x in str.split(',')]

