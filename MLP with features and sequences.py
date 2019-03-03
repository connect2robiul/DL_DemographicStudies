#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


print "============================================================"
print " MLP with features and sequences                            "
print "============================================================"
"""
        Number of all substrings in the string = n*(n+1)/2
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

X = np.empty((x.shape[0], x.shape[1]), dtype='float32')

for d in xrange(x.shape[1]):        
    xs = list(set(x[:, d]))
    if d == 0:
        print "Number of unique sequences =", len(xs), "\n"
    xd = {xs[i]: i for i in xrange(len(xs))}
    for l in xrange(x.shape[0]):
        X[l][d] = xd[x[l][d]]


X = StandardScaler().fit_transform(X)

SN_train = X[0:t, 0:1]          # sequences unique numbered
SN_test  = X[t:size, 0:1]

SF_train = X[0:t, 0:6]          # sequences and features
SF_test  = X[t:size, 0:6]

X_train = X[0:t, 1:]            # features only
X_test  = X[t:size, 1:]

y_train = y[0:t]
y_test  = y[t:size]

print "Train features  data shape =", X_train.shape, y_train.shape
print "Test  features  data shape =", X_test.shape,  y_test.shape
print "Train sequences data shape =", SN_train.shape, y_train.shape
print "Test  sequences data shape =", SN_test.shape,  y_test.shape
print "\nData preprocessing time, sec =  %0.2f" % (time.time() - start_time)

print "============================================================"

 
time_01 = time.time()

clf = MLPClassifier(hidden_layer_sizes=(200, 100, 100, ))

clf.fit(SN_train, y_train)
print clf
print "============================================================"
print "\nMLP classification by sequences" 

time_02 = time.time()
print "\tModel fitting time .... %0.2f" % (time_02 - time_01)


SN_score = clf.score(SN_test, y_test)

time_03 = time.time()
print "\tPrediction time ....... %0.2f" % (time_03 - time_02)
print "\tTotal time ............ %0.2f" % (time_03 - time_01)    
print '\tAccuracy (by sequences):  %0.3f' % SN_score

del clf    
print "============================================================"

    
print "\nMLP classification by features"    
time1 = time.time()

clf = MLPClassifier(hidden_layer_sizes=(200, 100, 100, ))

clf.fit(X_train, y_train)

time2 = time.time()
print "\tModel fitting time .... %0.2f" % (time2 - time1)

#F_y_proba = clf.predict_proba(X_test)

F_score = clf.score(X_test, y_test)

time3 = time.time()
print "\tPrediction time ....... %0.2f" % (time3 - time2)
print "\tTotal time ............ %0.2f" % (time3 - time1)    
print '\tAccuracy (by features):  %0.3f' % F_score

del clf    
print "============================================================"


print "\nMLP classification by sequences and features"    
time_1 = time.time()

clf = MLPClassifier(hidden_layer_sizes=(200, 100, 100, ))

clf.fit(SF_train, y_train)

time_2 = time.time()
print "\tModel fitting time .... %0.2f" % (time_2 - time_1)

#SF_y_proba = clf.predict_proba(SF_test)

SF_score = clf.score(SF_test, y_test)

time_3 = time.time()
print "\tPrediction time ....... %0.2f" % (time_3 - time_2)
print "\tTotal time ............ %0.2f" % (time_3 - time_1)    
print '\tAccuracy (by sequences and features):  %0.3f' % SF_score

del clf    
print "============================================================"
