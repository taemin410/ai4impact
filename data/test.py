import pandas as pd
df = pd.DataFrame([[1,2],[3,4],[5,6]], columns=list("ab"))

for i, row in df.iterrows():
    if row['a'] > 2:
        row = pd.DataFrame(row).transpose()
        print('Before\n',row,'\n####\n', df,'\n')
        print('!!!!!!!!!!\n', [df[:i+1], row, df[i+1:]],'\n\n\n')
        df = pd.concat([df[:i+1], row, df[i+1:]])
        print('After\n', df,'\n__________________')
print(df)