import pandas as pd

df1 = pd.read_csv("./data/alist_approx_20.txt")
df2 = pd.read_csv("./data/alist_approx_50.txt")
df3 = pd.read_csv("./data/80-sit-stand-usb-down.txt")
df4 = pd.read_csv("./data/no-gesture-usb-towards-hand.txt")
df5 = pd.read_csv("./data/no-points-usb-down.txt")
datas = [df1, df2, df3, df4, df5]

save_to_file = True  # only saves if True
print_scale_factors = True

for i, df in enumerate(datas):
    df_a = df[['ax', 'ay', 'az']]

    a_min = min(df_a.min())
    df_a -= a_min
    a_max = max(df_a.max())
    df_a /= a_max
    df[['ax', 'ay', 'az']] = df_a

    df_q = df[['qw', 'qx', 'qy', 'qz']]
    q_min = min(df_q.min())
    df_q -= q_min
    q_max = max(df_q.max())
    df_q /= q_max
    df[['qw', 'qx', 'qy', 'qz']] = df_q

    df_e = df[["e" + str(i) for i in range(8)]]
    e_min = min(df_e.min())
    df_e -= e_min
    e_max = max(df_e.max())
    df_e /= e_max
    df[["e" + str(i) for i in range(8)]] = df_e

    df_g = df['gesture']
    df_g = pd.DataFrame([1 if b is True else 0 for b in df_g])
    df['gesture'] = df_g

    if print_scale_factors:
        print("a {:f}:{:f}, q {:f}:{:f}, e {:f}:{:f}".format(a_min, a_max, q_min, q_max, e_min, e_max))

    if save_to_file:
        df.to_csv(path_or_buf="./processed-data/df-" + str(i) + ".csv")
