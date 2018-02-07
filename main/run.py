import pandas
from sklearn.neighbors import KNeighborsClassifier as k
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy
dataset= pandas.read_csv(r'/home/mihodihasan/Desktop/WithTarget.csv')

knn = k(n_neighbors=10)
#
feature=numpy.zeros(shape=(len(dataset),8))
feature[:,0]=dataset.latitude
feature[:,1]=dataset.longitude
feature[:,2]=dataset.date
feature[:,3]=dataset.month
feature[:,4]=dataset.year
feature[:,5]=dataset.hour
feature[:,6]=dataset.minute
feature[:,7]=dataset.day
target=numpy.zeros(len(dataset))
target=dataset.target
knn.fit(feature, dataset.target)

res = knn.predict(feature)
print('accuracy')
print(metrics.accuracy_score(dataset.target, res))

X_train, X_test, y_train, y_test = train_test_split(feature, dataset.target,test_size=.5)
knn.fit(X_train, y_train);
y_pred = knn.predict(X_test)
print('train_test_split')
print(metrics.accuracy_score(y_test, y_pred))