import pandas as pd

df1 = pd.read_csv('task1_test_stage.csv')
df2 = pd.read_csv('test_result2.csv')

df1['label'] = df2['label'].tolist()

df1.to_csv('test2.csv', sep=',', index=False)
