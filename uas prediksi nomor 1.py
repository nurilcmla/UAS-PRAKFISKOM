import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#Database

FileDB = 'uasbaru1.txt'
Database = pd.read_csv(FileDB, sep=" ", header=0)
print ("--------------------")
print (Database)
#x = data, y = target
x = Database[[u'x']] #ciri1, ciri2, dst
y = Database.Target

regr = LinearREgression().fit(x,y)
regr.score(x, y)

#Data uji
predict = np.array([[1100 ]])

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))

