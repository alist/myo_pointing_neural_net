import pandas as pd
import math

df0 = pd.read_csv("./data/alist_approx_20.txt")
df1 = pd.read_csv("./data/alist_approx_50.txt")
df2 = pd.read_csv("./data/80-sit-stand-usb-down.txt")
df3 = pd.read_csv("./data/no-gesture-usb-towards-hand.txt")
df4 = pd.read_csv("./data/no-points-usb-down.txt")
df5 = pd.read_csv("./data/wrist-exercise-usb-down-0g.txt")
df6 = pd.read_csv("./data/clap-dance-usb-down-0g.txt")
df7 = pd.read_csv("./data/50-or-more-gestures-light-usb-down.txt")
datas = [df0, df1, df2, df3, df4, df5, df6, df7]

universal_scale = True  # if true uses the following scale values
a_scale = (-10, 10)
q_scale = (-1 * math.pi, 3 * math.pi)
e_scale = (-128, 255)

save_to_file = True  # only saves if True
print_scale_factors = True

for i, df in enumerate(datas):
    df_a = df[['ax', 'ay', 'az']]

    a_min = a_scale[0] if universal_scale else min(df_a.min())
    df_a -= a_min
    a_max = a_scale[1] if universal_scale else max(df_a.max())
    df_a /= a_max
    df[['ax', 'ay', 'az']] = df_a

    df_q = df[['qw', 'qx', 'qy', 'qz']]
    q_min = q_scale[0] if universal_scale else min(df_q.min())
    df_q -= q_min
    q_max = q_scale[1] if universal_scale else max(df_q.max())
    df_q /= q_max
    df[['qw', 'qx', 'qy', 'qz']] = df_q

    df_e = df[["e" + str(i) for i in range(8)]]
    e_min = e_scale[0] if universal_scale else min(df_e.min())
    df_e -= e_min
    e_max = e_scale[1] if universal_scale else max(df_e.max())
    df_e /= e_max
    df[["e" + str(i) for i in range(8)]] = df_e

    df_g = df['gesture']
    df_g = pd.DataFrame([1 if b is True else 0 for b in df_g])
    df['gesture'] = df_g

    if print_scale_factors:
        if universal_scale: print("Using universal scale-- the following for reference: ")
        print("df{:d}: a {:f}:{:f}, q {:f}:{:f}, e {:f}:{:f}".format(i, a_min, a_max, q_min, q_max, e_min, e_max))

    if save_to_file:
        df.to_csv(path_or_buf="./processed-data/df-" + str(i) + ".csv")
