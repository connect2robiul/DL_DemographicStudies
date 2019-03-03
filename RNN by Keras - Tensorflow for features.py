#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from keras.backend import backend
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
#from keras.layers.embeddings import Embedding

print "============================================================"
print " RNN for demographic features          Muratova Anna        "
print "============================================================"


start_time = time.time()

df = pd.read_excel("table.xlsx")

df.dropna(inplace = True)   # remove empty (NaN) lines
df = shuffle(df, random_state=7)
print "Input data shape =", df.shape
size = df.shape[0]
t = int(size * 0.8)         # train part size

x = df.as_matrix()[:size, 0:6]
y = df.as_matrix()[:size, 6] 
del df

X = np.empty((x.shape[0], x.shape[1]), dtype='int32')

for d in xrange(x.shape[1]):        
    xs = list(set(x[:, d]))
    if d == 0:
        print "Number of unique sequences =", len(xs), "\n"
    xd = {xs[i]: i for i in xrange(len(xs))}
    for l in xrange(x.shape[0]):
        X[l][d] = xd[x[l][d]]

#X = StandardScaler().fit_transform(X)

XF_train = X[0:t, 1:]           # features only
XF_test  = X[t:size, 1:]
#print XF_train[0:10]

yn = np.empty((size), dtype='int32')

ys = list(set(y[:]))
yd = {ys[i]: i for i in xrange(len(ys))}
for l in xrange(len(yn)):
        yn[l] = yd[y[l]]


y_train = yn[0:t]
y_test  = yn[t:size]

# all unique characters to the set
events = set()
for seq in x[:, 0]:
    for event in seq:
        events.add(event)

events = list(events)
#print events

event_to_id = {t:i+1 for i,t in enumerate(events)}
#print event_to_id

max_seq_len = 8
seq_events_numbered = np.zeros((x.shape[0], max_seq_len), dtype='int32')

for i in xrange(seq_events_numbered.shape[0]):
    for k in xrange(len(x[i][0])):
        seq_events_numbered[i][k] = event_to_id[x[i][0][k]]

S_train = seq_events_numbered[0:t, :]       # train sequences
S_test  = seq_events_numbered[t:size, :]    # test  sequences


print "Train features  data shape =", XF_train.shape, y_train.shape
print "Test  features  data shape =", XF_test.shape,  y_test.shape
print "\nData preprocessing time, sec =  %0.2f" % (time.time() - start_time)

print "============================================================"

print "Keras backend =", backend()
 
time_01 = time.time()
print "\nRNN classification by features" 

features = Sequential()
features.add(Dense(100, activation="sigmoid", input_dim=5))
features.add(Dropout(0.1))
features.add(Dense(100, activation='sigmoid'))
features.add(Dense(1, activation="sigmoid"))
features.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
features.fit(XF_train, y_train, epochs=400, batch_size=100)

time_02 = time.time()
score, acc = features.evaluate(XF_test, y_test, verbose=0)

time_03 = time.time()

print "\tModel fitting time .... %0.2f" % (time_02 - time_01)
print "\tPrediction time ....... %0.2f" % (time_03 - time_02)
print "\tTotal time ............ %0.2f" % (time_03 - time_01)    
print("Accuracy: %.3f" % (acc))
print "============================================================"
