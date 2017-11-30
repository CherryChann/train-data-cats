import csv
import nltk 
import numpy
tokenized_list = []
with open('task-assign.csv', 'r') as csvfile:  # this will close the file automatically.
    reader = csv.reader(csvfile)
    for row in reader:
        for field in row:
            tokens = nltk.word_tokenize(field)
            tokenized_list.append(tokens)
with open("tokenize-data.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for word in tokenized_list:
        wr.writerow(word)