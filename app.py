import csv
import nltk 
import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression

sentences = []
f = open('sentences.csv', 'r')
for line in f:
    words = line.strip().split(',')
    sentences.append(words)

uniquewords = []
f = open('uniquewords.csv', 'r')
for line in f:
    uniquewords.append(line.strip())

labels = []
data = []

for sentence in sentences:
    inner_data = []
    for word in uniquewords:
        if word in sentence:
            inner_data.append('1')
        else:
            inner_data.append('0')
    data.append(inner_data)

for i in range(0, 40):
    inner_label = []
    if i < 20:
        inner_label.append(0)
        inner_label.append(1)
    else:
        inner_label.append(1)
        inner_label.append(0)
    labels.append(inner_label)

labels = np.array(labels, dtype=np.float32)
data = np.array(data, dtype=np.float32)

print labels

from tflearn.data_utils import shuffle
data, labels = shuffle(data, labels)

network = input_data(shape=[None, 88])

# network = conv_1d(network, 88, 1, activation='relu')
# network = max_pool_1d(network, 2)

# network = conv_1d(network, 88*2, 1, activation='relu')
# network = max_pool_1d(network, 2)

network = fully_connected(network, 88, activation='relu')
network = fully_connected(network, 88*2, activation='relu')
network = fully_connected(network, 88, activation='relu')
network = dropout(network, 0.5)

network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

model = tflearn.DNN(network)
model.fit(data, labels, n_epoch=10, shuffle=True, validation_set=None, show_metric=True, batch_size=None, snapshot_epoch=True, run_id='task-classifier')
model.save("task-classifier.tfl")
print("Network trained and saved as task-classifier.tfl")


sentence_one = ['can', 'you', 'prepare', 'task', 'for', 'smartcats', 'in', 'the', 'workflow']
sentence_two = ['i', 'cannot', 'sorry', 'i', 'am', 'busy']

vector_one = []
for word in uniquewords:
    if word in sentence_one:
        vector_one.append(1)
    else:
        vector_one.append(0)

vector_two = []
for word in uniquewords:
    if word in sentence_two:
        vector_two.append(1)
    else:
        vector_two.append(0)

vector_one = np.array(vector_one, dtype=np.float32)
vector_two = np.array(vector_two, dtype=np.float32)


label = model.predict_label([vector_one, vector_two])
print(label)

pred = model.predict([vector_one, vector_two])
print(pred)
print("vector:", pred[0][1])
print("vector:", pred[1][1])
