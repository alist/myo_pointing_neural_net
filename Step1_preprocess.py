import pandas as pd

df1 = pd.read_csv("./data/alist_approx_20.txt")
df2 = pd.read_csv("./data/alist_approx_50.txt")
datas = [df1, df2]

for i, df in enumerate(datas):
    df_a = df[['ax', 'ay', 'az']]
    df_a -= min(df_a.min())
    df_a /= max(df_a.max())
    df[['ax', 'ay', 'az']] = df_a

    df_q = df[['qw', 'qx', 'qy', 'qz']]
    df_q -= min(df_q.min())
    df_q /= max(df_q.max())
    df[['qw', 'qx', 'qy', 'qz']] = df_q

    df_e = df[["e" + str(i) for i in range(8)]]
    df_e -= min(df_e.min())
    df_e /= max(df_e.max())
    df[["e" + str(i) for i in range(8)]] = df_e

    df_g = df['gesture']
    df_g = pd.DataFrame([1 if b is True else 0 for b in df_g])
    df['gesture'] = df_g

    df.to_csv(path_or_buf="./processed-data/df-" + str(i) + ".csv")