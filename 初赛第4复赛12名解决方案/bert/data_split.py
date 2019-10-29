import sys
import pandas as pd
import numpy as np
import csv
import pprint

def main(K=10):

    f = open("./task1/train.csv")
    reader = csv.reader(f)
    data_lines = [row for row in reader]

    now = 1
    interval = int(len(data_lines) / K)
    df_splited = []
    for idx in range(0,K):
        if idx == K-1:
            df_splited.append(data_lines[now:])
        else:
            df_splited.append(data_lines[now:now+interval])
        now += interval

    assert len(df_splited) == K

    for idx in range(0,K):

        df_train = pd.DataFrame(np.concatenate(tuple([dfn for index, dfn in enumerate(df_splited) if index != idx]), axis=0), columns=['id','text','label'])
        pprint.pprint(df_train)
        print ("dataset train {} has shape {}.".format(idx+1, df_train.shape))

        df_dev = pd.DataFrame(df_splited[idx], columns=['id','text','label'])
        print ("dataset dev {} has shape {}.".format(idx+1, df_dev.shape))

        df_train.to_csv("./task1/train_{}.csv".format(idx+1), sep=',',index=False)
        df_dev.to_csv("./task1/dev_{}.csv".format(idx+1), sep=',',index=False)

if __name__ == "__main__":
    main(int(sys.argv[1]))
