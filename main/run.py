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

X_train, X_test, y_train, y_test = train_test_split(feature, dataset.target,test_size=.3)
knn.fit(X_train, y_train);
y_pred = knn.predict(X_test)
print('train_test_split')
print(metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(feature,target)
res=linreg.predict(feature)
print('ln reg')
print((100-metrics.mean_squared_error(dataset.target, res)))

linreg.fit(X_train, y_train);
y_pred = linreg.predict(X_test)
print('train_test_split  lin reg')
print(100-metrics.mean_squared_error(y_test, y_pred))


from sklearn import svm

svm_clf = svm.SVC()

svm_clf.fit(feature,target)

res=svm_clf.predict(feature)
print('svm')
print(metrics.accuracy_score(target,res))
print('train test split svm')
y_pred=svm_clf.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))