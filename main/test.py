import pandas as pd
import numpy as np
from sodapy import Socrata as repo
import time
import datetime
from sklearn.neighbors import KNeighborsClassifier as k
from sklearn import metrics
from sklearn.model_selection import train_test_split

results_df = pd.read_csv(r'/home/mihodihasan/dataset.csv')

unique_max = results_df['asset'].value_counts().idxmax()  # to determine which vehicle has maximum value
results_df = results_df[results_df['asset'] == unique_max]  # filtering to one vehicle data
results_df.to_csv(r'/home/mihodihasan/thesisDataset.csv',sep=',',encoding='utf-8')