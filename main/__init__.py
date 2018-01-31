import pandas as pd
import numpy as np
from sodapy import Socrata as repo

# Example authenticated client (needed for non-public datasets):
client = repo("data.cincinnati-oh.gov", "t1J7xReA9Slki6fndZ3ocYROj", "mihodihasan@gmail.com", "%&!Mhl0Mb%&!")

results = client.get("w2ka-rfbi", limit=80000)

results_df = pd.DataFrame.from_records(results)
unique_max = results_df['asset'].value_counts().idxmax()  # to determine which vehicle has maximum value
results_df = results_df[results_df['asset'] == str(unique_max)]  # filtering to one vehicle data
data_size = len(results_df)
location = np.zeros(data_size)
inner_index = 0
outer_index = 0
lat_threshold = .001
lng_threshold = .01
# while outer_index<data_size:
while outer_index < data_size:
    lat = results_df.iloc[outer_index]['latitude']
    lng = results_df.iloc[outer_index]['longitude']
    count = 0
    inner_index=0
    while inner_index < outer_index:
        if (abs(float(results_df.iloc[inner_index]['latitude']) - float(lat)) <= lat_threshold and abs(
                    float(results_df.iloc[inner_index]['longitude']) - float(lng)) <= lng_threshold):
            count += 1
            if (count >= 5):
                break
        inner_index += 1
    if (count >= 5):
        location[outer_index] = 1
    elif (count < 5):
        location[outer_index] = 0
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
target = location
#
feature = np.zeros(shape=(data_size, 9))
feature[:, 0] = date
feature[:, 1] = month
feature[:, 2] = year
feature[:, 3] = day
feature[:, 4] = hour
feature[:, 5] = minute
feature[:, 6] = second
feature[:, 7] = longitude
feature[:, 8] = latitude
#
# print(feature.columns)
# feature_array=pd.DataFrame.as_matrix(feature)
# print(pd.DataFrame.as_matrix(feature))
# target_array=pd.DataFrame.as_matrix(target)
from sklearn.neighbors import KNeighborsClassifier as k

knn = k(n_neighbors=10)
#
knn.fit(feature, target)

from sklearn import metrics

res = knn.predict(feature)
print('accuracy')
print(metrics.accuracy_score(target, res))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feature, target)
knn.fit(X_train, y_train);
y_pred = knn.predict(X_test)
print('train_test_split')
print(metrics.accuracy_score(y_test, y_pred))
