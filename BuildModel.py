import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv3D, MaxPool3D, LSTM, LSTMCell
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model

import numpy as np
import pandas as pd

num_classes = 2
validation_split = 0.8
number_data_columns = 16

batch_size = 64
epochs = 22

df = pd.read_csv("./data/joint.csv")
drop_last = len(df) % batch_size
df = df[:-1 * drop_last]

x_data = np.array(df.drop(columns=['time', 'gesture']))
x_data = np.expand_dims(x_data, axis=1)  # they want [[[1, 2, 3, ]], ... [[3,2,1, ...]
df_g = df['gesture']
y_data = np.array([[1, 0] if b == 1 else [0, 1] for b in df_g])

split_point = round(len(x_data) * validation_split)
split_point = split_point - (split_point % batch_size)  # need train to be %=0 by batch_size
x_test = x_data[split_point:]
y_test = y_data[split_point:]
x_train = x_data[:split_point]
y_train = y_data[:split_point]

model = Sequential()
model.add(LSTM(units=16,
               activation='relu',
               recurrent_activation='hard_sigmoid',
               input_shape=(1, number_data_columns),
               stateful=True,
               batch_size=batch_size))
model.add(Dense(units=32))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# TODO: How to make validation greatly penalize false positives and not care about false negatives?
# TODO: What does LSTM units do? How does the temporal aspect get carried along?
# TODO: Still, how do you quantify: A gesture happened in this window sometime?
# TODO: Also, what should I record as a gesture in order to make it easier for training?

