import pandas as pd
import os
from datetime import datetime as dt

df = pd.read_csv('./wind_energy_v2.csv')
out = df['time'].apply(lambda x : x[:-9]).value_counts()
print(out[-20:])

# root = './tmp/'
# dirs2_ = os.listdir(root)
dirs_ = os.listdir('./tmp')
# version_1 = [a for a in dirs_ if '-b' in a]
# version_2 = [a for a in dirs_ if '-b' not in a]

df = pd.read_csv('./history_cleaned/angerville-1.csv')
out1 = df['Time'].apply(lambda x : x[:-9]).value_counts()
print(out1[-5:], out1[:5], "\n\n_______________________\n\n")

df = pd.read_csv('./history_cleaned/angerville-1-b.csv')
out1 = df['Time'].apply(lambda x : x[:-9]).value_counts()
print(out1[-5:], out1[:5], "\n\n_______________________\n\n")


df = pd.read_csv('./tmp/angerville-1.csv')
out1 = df['Time'].apply(lambda x : x[:-9]).value_counts()
print(out1[-5:], out1[:5], "\n\n_______________________\n\n")

df = pd.read_csv('./tmp/angerville-1-b.csv')
out1 = df['Time'].apply(lambda x : x[:-9]).value_counts()
print(out1[-5:], out1[:5], "\n\n_______________________\n\n")


# for dir_ in dirs_:
#         tmp = pd.read_csv(root+ dir_)
#         added = 0
#         for i, row in tmp[:-1].iterrows():
#                 row = pd.DataFrame(row).transpose()
#                 time_diff = int((tmp['Time'][i+1] - tmp['Time'][i])/np.timedelta64(1,'D') / 0.25) -1 
#                 if time_diff != 0:
#                         new_row = []
#                         for j in range(time_diff):
#                         row['Time'] = row['Time'].apply( lambda x : x + datetime.timedelta(hours=6))
#                         new_row.append(row.copy())
#                         tmp = pd.concat([tmp[:i+added+1]] + new_row + [tmp[i+added+1:]])
#                         added += time_diff                        
#         tmp.to_csv('../data/tmp/'+dir_[dir_.rfind('/'):])

