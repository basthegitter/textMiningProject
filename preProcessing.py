import csv
import pandas as pd

'''
def read_data(name):
    with open(name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        descriptions = []
        genres = []
        for row in readCSV:
           #print(row)
            genres.append(row[2])
            descriptions.append(row[3])

    return descriptions, genres
'''
imdbdata=pd.read_csv('../input/IMDB-Movie-Data.csv')
#d, g = read_data('IMDB-Movie-Data.csv')
#str = g[1]
#print str
#[x.strip() for x in str.split(',')]

print str
