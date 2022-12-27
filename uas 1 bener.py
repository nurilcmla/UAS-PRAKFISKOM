import numpy as np
import csv
import time
from sklearn import svm
import pandas as pd

#Database
# x = Data, y = Target
#x = [[1],[3],[5],[7],[9]]
#y = [2, 6,10, 14, 18]
FileDB = 'uasbaru1.txt'
Database = pd.read_csv(FileDB, sep=" ", header=0)
print ("--------------------")
print (Database)
#x = data, y = target
x = Database[[u'x']] #ciri1, ciri2, dst
y = Database.Target

clf = svm.svc()
clf.fit(x.values,y)

print(clf.predict( [[(x**15)+3]] ))
