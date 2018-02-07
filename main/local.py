import pandas as pd
import numpy as np
from sodapy import Socrata as repo
import time
import datetime
from sklearn.neighbors import KNeighborsClassifier as k
from sklearn import metrics
from sklearn.model_selection import train_test_split

start_time = time.time()


results_df = pd.read_csv(r'/home/mihodihasan/thesisDataset.csv')

data_size = len(results_df)
print('total data{}'.format(results_df.shape))
location = np.zeros(data_size)
inner_index = 0
outer_index = 0
lat_threshold = .001
lng_threshold = .01
# while outer_index<data_size:
while outer_index < data_size:
    print('outer index is {}'.format(outer_index))
    lat = results_df.iloc[outer_index]['latitude']
    lng = results_df.iloc[outer_index]['longitude']
    count = 0
    inner_index = 0
    while inner_index < outer_index:
        print('inner index is {}'.format(inner_index))
        if (abs(float(results_df.iloc[inner_index]['latitude']) - float(lat)) <= lat_threshold and abs(
                    float(results_df.iloc[inner_index]['longitude']) - float(lng)) <= lng_threshold):
            count += 1
            if count >= 5:
                break
        inner_index += 1
    if count >= 5:
        location[outer_index] = 1
    elif count < 5:
        location[outer_index] = 0
    print('total count {}'.format(count))
    outer_index += 1
# unique, counts = np.unique(location, return_counts=True)
#
# print(np.asarray((unique, counts)).T)

idx = 0
date = np.zeros(data_size)
year = np.zeros(data_size)
month = np.zeros(data_size)
day = np.zeros(data_size)
hour = np.zeros(data_size)
minute = np.zeros(data_size)
# second = np.zeros(data_size)
for i in results_df['time']:
    year[idx] = i.split(' ')[0].split('/')[2]
    month[idx] = i.split(' ')[0].split('/')[0]
    date[idx] = i.split(' ')[0].split('/')[1]
    hour[idx] = i.split(' ')[1].split(':')[0]
    minute[idx] = i.split(' ')[1].split(':')[1]
    # second[idx] = i.split(' ')[1].split('.')[0].split(':')[2]
    # print(i.split('T')[1].split('.')[0].split(':'))
    # print (date)


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
target = location
#
feature = np.zeros(shape=(data_size, 8))
feature[:, 0] = date
feature[:, 1] = month
feature[:, 2] = year
feature[:, 3] = day
feature[:, 4] = hour
feature[:, 5] = minute
# feature[:, 6] = second
feature[:, 6] = longitude
feature[:, 7] = latitude
#
# print(feature.columns)
# feature_array=pd.DataFrame.as_matrix(feature)
# print(pd.DataFrame.as_matrix(feature))
# target_array=pd.DataFrame.as_matrix(target)
knn = k(n_neighbors=10)
#
knn.fit(feature, target)

res = knn.predict(feature)
print('accuracy')
print(metrics.accuracy_score(target, res))

X_train, X_test, y_train, y_test = train_test_split(feature, target)
knn.fit(X_train, y_train);
y_pred = knn.predict(X_test)
print('train_test_split')
print(metrics.accuracy_score(y_test, y_pred))
print("time elapsed: {:.2f}s".format(time.time() - start_time))

temp_df=pd.DataFrame({'date':date,'month':month,'year':year,'day':day,'hour':hour,'minute':minute,'latitude':latitude,'longitude':longitude,'target':location})
temp_df.to_csv('/home/mihodihasan/Desktop/WithTarget.csv',sep=',',encoding='utf-8')