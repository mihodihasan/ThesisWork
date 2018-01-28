import pandas as pd
import numpy as np
from sodapy import Socrata as repo

# Example authenticated client (needed for non-public datasets):
client = repo("data.cincinnati-oh.gov", "t1J7xReA9Slki6fndZ3ocYROj", "mihodihasan@gmail.com", "%&!Mhl0Mb%&!")

results = client.get("w2ka-rfbi", limit=80000)

results_df = pd.DataFrame.from_records(results)
data_size = len(results_df)
# print(results_df.asset.drop_duplicates('first'))
# print(results_df.longitude.size)
idx = 0
date = np.zeros(data_size)
year = np.zeros(data_size)
month = np.zeros(data_size)
day = np.zeros(data_size)
hour = np.zeros(data_size)
minute = np.zeros(data_size)
second = np.zeros(data_size)
for i in results_df['time']:
    year[idx] = i.split('T')[0].split('-')[0]
    month[idx] = i.split('T')[0].split('-')[1]
    date[idx] = i.split('T')[0].split('-')[2]
    hour[idx] = i.split('T')[1].split('.')[0].split(':')[0]
    minute[idx] = i.split('T')[1].split('.')[0].split(':')[1]
    second[idx] = i.split('T')[1].split('.')[0].split(':')[2]
    # print(i.split('T')[1].split('.')[0].split(':'))
    # print (date)
    import datetime

    day[idx] = datetime.datetime(int(year[idx]), int(month[idx]), int(date[idx])).weekday()

    idx = idx + 1
# day_series = pd.Series(day)
# day_series.name = 'Day'
# date_series = pd.Series(date.tolist())
# date_series.name = 'Date'
# time_series = pd.Series(time.tolist())
# time_series.name = 'Time'

latitude = pd.Series(results_df.latitude).values
longitude = pd.Series(results_df.longitude).values
target = np.zeros(data_size)
target = latitude
#
feature = np.zeros(shape=(data_size, 8))
feature[:, 0] = date
feature[:, 1] = month
feature[:, 2] = year
feature[:, 3] = day
feature[:, 4] = hour
feature[:, 5] = minute
feature[:, 6] = second
feature[:, 7] = longitude
#
# print(feature.columns)
# feature_array=pd.DataFrame.as_matrix(feature)
# print(pd.DataFrame.as_matrix(feature))
# target_array=pd.DataFrame.as_matrix(target)
from sklearn.neighbors import KNeighborsClassifier as k

knn = k(n_neighbors=2)
#
# knn.fit(feature, target)

from sklearn import metrics

# res = knn.predict(feature)
# print(metrics.accuracy_score(target, res))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1, random_state=0)
knn.fit(X_train,y_train);
y_pred=knn.predict(X_test)

# print(metrics.accuracy_score(y_test, y_pred))
print(results_df)