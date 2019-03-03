#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from keras.backend import backend
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, GRU
from keras.layers.embeddings import Embedding


print "============================================================"
print " RNN for demographic sequences          Muratova Anna       "
print "============================================================"

"""    
   Choose (uncomment) one line for the RNN: SimpleRNN, GRU or LSTM
"""

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


print "Train sequences data shape =", S_train.shape, y_train.shape
print "Test  sequences data shape =", S_test.shape,  y_test.shape
print "\nData preprocessing time, sec =  %0.2f" % (time.time() - start_time)

print "============================================================"

print "Keras backend =", backend()
 
time_01 = time.time()
print "\nRNN classification by sequences" 


seq = Sequential()
seq.add(Embedding(input_dim=9, output_dim=200, input_length=8))
"""
    Uncomment one line for the RNN: SimpleRNN, GRU or LSTM
"""
seq.add(SimpleRNN(200))
#seq.add(LSTM(200, recurrent_dropout=0.05))
#seq.add(GRU(200, recurrent_dropout=0.05)
seq.add(Dense(1, activation='sigmoid'))

seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(seq.summary())
seq.fit(S_train, y_train, epochs=200, batch_size=200)

time_02 = time.time()

score, acc = seq.evaluate(S_test, y_test, verbose=0)

time_03 = time.time()

print "\tModel fitting time .... %0.2f" % (time_02 - time_01)
print "\tPrediction time ....... %0.2f" % (time_03 - time_02)
print "\tTotal time ............ %0.2f" % (time_03 - time_01)    
print("Accuracy: %.3f" % (acc))
  
print "============================================================"
