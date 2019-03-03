#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

print "============================================================"
print " SVM with features and custom kernel for sequences          "
print "============================================================"
"""
        Number of all substrings in the string = n*(n+1)/2
"""

def sim_acs(S, T):      
    """All Common Substrings similarity (ACS)"""
    m = len(S)
    n = len(T)
    count = 0
    for i in xrange(m):
        j = 1           # substring length
        while i + j <= m and j <= n:     
            k = T.find(S[i:i+j])    
            if k >= 0:
                count += 1

            j = j + 1
            
    l = max(m, n)
    #print acs_set
    return np.float32(count) / (l*(l+1)/2)

                 
def sim_lcs(S, T):      
    """Longest Common Substring similarity (LCS)"""
    m = len(S)
    n = len(T)
    lcs = 0             # longest substring length
    for i in xrange(m):
        j = 1           # substring length
        while i + j <= m and j <= n:     
            k = T.find(S[i:i+j])
            if k >= 0:
                if lcs < j:
                    lcs = j

            j = j + 1
                       
    return np.float32(lcs) / max(m, n)

   
def sim_prefix(S, T):               
    """Common Prefix Similarity (CPS)"""
    m = len(S)
    n = len(T)
    l = max(m, n)
    if l == 0:
        return 0.
        
    minmn = min(m, n)
    k = 0
    while k < minmn:
        if S[k] == T[k]:
            k = k + 1
        else:
            break
    # k = common prefix length
    return np.float32(float(k) / float(l))
    
    
def kernel(s, t, sim_function):  # Gram matrix creation for any similarity
    m = s.shape[0]
    n = t.shape[0]
    X = np.empty((m, n), dtype='float32')
    for i in xrange(m):
        for j in xrange(n):
            X[i][j] = sim_function(s[i], t[j])

    return X    

"""    
   Input data preprocessing
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


S_train = x[0:t, 0]             # sequences for custom kernels
S_test  = x[t:size, 0]

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
print "Train sequences data shape =", S_train.shape, y_train.shape
print "Test  sequences data shape =", S_test.shape,  y_test.shape
print "\nData preprocessing time, sec =  %0.2f" % (time.time() - start_time)

print "============================================================"

    
print "\nSVM classification by features"    
time1 = time.time()

clf = svm.SVC(probability=True)

clf.fit(X_train, y_train)

time2 = time.time()
print "\tModel fitting time .... %0.2f" % (time2 - time1)

F_y_proba = clf.predict_proba(X_test)

F_score = clf.score(X_test, y_test)

time3 = time.time()
print "\tPrediction time ....... %0.2f" % (time3 - time2)
print "\tTotal time ............ %0.2f" % (time3 - time1)    
print '\tAccuracy (by features):  %0.3f' % F_score
#print '\tAccuracy:  %0.3f' % accuracy_score(y_test, y_pred)
del clf    
print "============================================================"


print "\nSVM classification by sequences"    
time_1 = time.time()

clf = svm.SVC(probability=True)

clf.fit(SN_train, y_train)

time_2 = time.time()
print "\tModel fitting time .... %0.2f" % (time_2 - time_1)

SN_y_proba = clf.predict_proba(SN_test)

SN_score = clf.score(SN_test, y_test)

time_3 = time.time()
print "\tPrediction time ....... %0.2f" % (time_3 - time_2)
print "\tTotal time ............ %0.2f" % (time_3 - time_1)    
print '\tAccuracy (by sequences):  %0.3f' % SN_score
#print '\tAccuracy:  %0.3f' % accuracy_score(y_test, y_pred)
del clf    
print "============================================================"

   
for similarity in [sim_prefix, sim_acs, sim_lcs]:
    """
    Classification with custom kernels 
    for different sequences similarity functions
    """
    print "\n", similarity.func_doc # func_name
    time1 = time.time()
    
    clf = svm.SVC(kernel='precomputed', probability=True, cache_size=1000)
    
    kernel_train = kernel(S_train, S_train, sim_function = similarity)
        
    clf.fit(kernel_train, y_train)
    
    time2 = time.time()
    print "\tModel fitting time .... %0.2f" % (time2 - time1)
    
    kernel_test = kernel(S_test, S_train, sim_function = similarity)
    # The kernel for test by column 0 (sequences) 

    S_y_proba = clf.predict_proba(kernel_test)
    #y_pred = clf.predict(kernel_test)
    S_score = clf.score(kernel_test, y_test)
    
    time3 = time.time()
    print "\tPrediction time ....... %0.2f" % (time3 - time2)
    print "\tTotal time ............ %0.2f" % (time3 - time1)    
    print '\tAccuracy (by custom kernel):  %0.3f' % S_score
    #print '\tAccuracy:  %0.3f' % accuracy_score(y_test, y_pred)
    
    del clf, kernel_train, kernel_test
    
    """    
    Calculate integrated classifiation using probabilities 
    from custom kernel for sequences and 
    from SVM default classification for features
    """    
    y_pred = np.empty_like(y_test)
    
    first_corrections  = 0
    second_corrections = 0
    
    for i in xrange(F_y_proba.shape[0]):
        #if F_y_proba[i][0] + S_y_proba[i][0] > 1:
        if (F_score * F_y_proba[i][0] + S_score * S_y_proba[i][0]) / (F_score + S_score) > 0.5:
            if y_test[i] == "m":
                first_corrections += 1
            y_pred[i] = "f"
        else:        
            if y_test[i] == "f":
                second_corrections += 1
            y_pred[i] = "m"
            
    print "\nCorrections by features: ", first_corrections + second_corrections            
    print 'Accuracy (by custom kernel and features):  %0.3f' % accuracy_score(y_test, y_pred)   
    print "============================================================"


print "\nSVM classification by sequences and features"    
time4 = time.time()

clf = svm.SVC(probability=True)

clf.fit(SF_train, y_train)

time5 = time.time()
print "\tModel fitting time .... %0.2f" % (time5 - time4)

SF_y_proba = clf.predict_proba(SF_test)

SF_score = clf.score(SF_test, y_test)

time6 = time.time()
print "\tPrediction time ....... %0.2f" % (time6 - time5)
print "\tTotal time ............ %0.2f" % (time6 - time4)    
print '\tAccuracy (by sequences and features):  %0.3f' % SF_score
#print '\tAccuracy:  %0.3f' % accuracy_score(y_test, y_pred)
del clf    
print "============================================================"
