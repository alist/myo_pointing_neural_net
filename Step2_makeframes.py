import numpy as np
import pandas as pd
np.set_printoptions(threshold='nan')

df1 = pd.read_csv("./processed-data/df-0.csv")
df2 = pd.read_csv("./processed-data/df-1.csv")
datas = [df1, df2]

#  Make a picture out of the last 60 (Nf) data-points.
#  Steps:
# 1. Starting with dp 60 through dp N
# 2. Select dp i-60 through i
# 3. If more than 50% of the labels are 'gesture', label entire picture as gesture
# Save out the picture set

size_of_frame = 60
gesture_threshold = 0.5  # in percent

framed_dfs = []

write_out_each = True
write_out_concat = True
pre_balance = True  # Decides whether to drop non-gesture frames till we have equal gesture and non-gesture frames

for i_df, df in enumerate(datas):
    frames = []
    for i in range(size_of_frame, len(df)):
        frame_points = df.loc[range(i - size_of_frame, i)]
        gesture_count = sum(frame_points['gesture'])
        is_gesture = gesture_count >= size_of_frame * gesture_threshold
        frame_points.drop(columns=['gesture', 'time'], inplace=True)
        new_row = (frame_points.values, 1 if is_gesture else 0)
        frames.append(new_row)
    if len(frames) < size_of_frame:
        continue
    df_framed: pd.DataFrame = pd.DataFrame(data=frames)
    framed_dfs.append(df_framed)
    if write_out_each:
        df_values: np.ndarray = df_framed.values
        np.save("./processed-data/df-framed-" + str(i_df) + ".npy", arr=df_values, allow_pickle=True)

df_cat = pd.concat(framed_dfs)
df_cat.reset_index(inplace=True, drop=True)

if pre_balance:
    number_gesture = len(df_cat[df_cat[1] == 1])
    number_normal = len(df_cat) - number_gesture
    number_to_drop = number_normal - number_gesture

    for i in reversed(range(len(df_cat))):
        if df_cat.iloc[i][1] == 0:
            df_cat.drop(i, inplace=True)
            number_to_drop = number_to_drop - 1
            if number_to_drop is 0:
                break

if write_out_concat:
    np.save(file="./processed-data/df-framed-concat" + ("-balanced" if pre_balance else "") + ".npy",
            arr=df_cat.values,
            allow_pickle=True)
