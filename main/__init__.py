import pandas as pd
from sodapy import Socrata as repo

# Example authenticated client (needed for non-public datasets):
client = repo("data.cincinnati-oh.gov", "t1J7xReA9Slki6fndZ3ocYROj", "mihodihasan@gmail.com", "%&!Mhl0Mb%&!")

results = client.get("w2ka-rfbi", limit=70000)

results_df = pd.DataFrame.from_records(results)

# print(results_df.asset.drop_duplicates('first'))
# print(results_df.longitude.size)
idx = 0
date = {}
day = {}
for i in results_df['loadts']:
    date[idx] = i.split('T')[0].split('-')
    # print (date)
    import datetime

    day_no = datetime.datetime(int(date[idx][0]), int(date[idx][1]), int(date[idx][2])).weekday()
    if day_no == 0:
        day[idx] = 'Monday'
    elif day_no == 1:
        day[idx] = 'Tuesday'
    elif day_no == 2:
        day[idx] = 'Wednesday'
    elif day_no == 3:
        day[idx] = 'Thursday'
    elif day_no == 4:
        day[idx] = 'Friday'
    elif day_no == 5:
        day[idx] = 'Saturday'
    elif day_no == 6:
        day[idx] = 'Sunday'

    # datesArray=s.split('T')[0].split('-')
    # print(datesArray)
    idx = idx + 1
day_series = pd.Series(day)

target = pd.DataFrame.from_items(
    [(results_df.latitude.name, results_df.latitude), (results_df.longitude.name, results_df.longitude)])

feature = pd.DataFrame.from_items(
    [(results_df.latitude.name, results_df.latitude), (results_df.longitude.name, results_df.longitude)])
# print(target)
print(type(day_series))
print(len(day_series))
print(day_series)
