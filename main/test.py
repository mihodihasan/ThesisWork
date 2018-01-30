import pandas as pd
import numpy as np
from sodapy import Socrata as repo

client = repo("data.cincinnati-oh.gov", "t1J7xReA9Slki6fndZ3ocYROj", "mihodihasan@gmail.com", "%&!Mhl0Mb%&!")

results = client.get("w2ka-rfbi", limit=80000)

results_df = pd.DataFrame.from_records(results)
# unique=pd.Series.unique(results_df)
# unique=results_df.unique()
u=results_df['asset'].value_counts().idxmax()
results_df=results_df[results_df['asset'] == str(u)]

print(results_df)
