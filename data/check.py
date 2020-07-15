import pandas as pd
import os
from datetime import datetime as dt

dirs_ = os.listdir('./history_cleaned')
version_1 = [a for a in dirs_ if '-b' in a]
version_2 = [a for a in dirs_ if '-b' not in a]

for v1, v2 in zip(version_1, version_2):
    df1 = pd.read_csv('./history_cleaned/'+ v1)
    df2 = pd.read_csv('./history_cleaned/'+ v2)

    df1['Time'] = df1['Time'].apply(lambda x : x[:-9]).value_counts()
    df2['Time'] = df2['Time'].apply(lambda x : x[:-9]).value_counts()

    print('____________________\n\n',dir_,type(out), out[out<4].index)

#     df['Time'] = df['Time'].apply(lambda x : dt.strptime(x[2:-3]+":00", '%y/%m/%d %H:%M:%S'))
#     # out = df['Time'].apply(lambda x : x[:-9]).value_counts()
#     # print(type(out), out[-20:])
#     for i, row in df.iterrows():
#         if i % 4 == 0:
#             if row['Time'].hour != 0:
#                 try:
                    
#         break
#     break
        # df.to_csv('./final_history/'+dir_)

# print()