import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential, load_model
from sklearn.utils import class_weight
# from sklearn.preprocessing import LabelEncoder  # if we had labels other than integers

import numpy as np

num_classes = 2
balance_class_weights = True
validation_split = 0.25  # This time we're taking from front
number_data_columns = 16
frame_size = 60

batch_size = 32
epochs = 15

lstm_units = 20

print("frame_size {:d}, validation_split {:f}, batch_size {:d}, epochs {:d}, lstm_units {:d}"
      .format(frame_size, validation_split, batch_size, epochs, lstm_units))

# Data where col 0 is the data picture, and col 1 is the label
LOAD_CAT_BALANCED = "./processed-data/df-framed-concat-balanced.npy"
LOAD_DFS = ["./processed-data/df-framed-0.npy", "./processed-data/df-framed-1.npy"]
data_set = LOAD_DFS

print("dataset: " + str(data_set))

df: np.ndarray = None

if data_set is LOAD_DFS:
    df = np.concatenate([np.load(file) for file in LOAD_DFS])
else:
    df = np.load(LOAD_CAT_BALANCED)

df = df[0: len(df) - len(df) % batch_size]  # trim off non-batchable

data_column = 0
gesture_classification_column = 1

x_data = np.stack(df[:, data_column])
x_data = x_data.reshape(x_data.shape + tuple([1]))  # if using convo and maxpool vs lstm
df_g = df[:, gesture_classification_column]
y_data = df_g

class_weights = [1.0/num_classes for _ in range(num_classes)]
if balance_class_weights:
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_data), y_data)

# we want output vectors as np arrays
y_data = keras.utils.to_categorical(df_g, num_classes)

split_point = round(len(x_data) * validation_split)
split_point = split_point - (split_point % batch_size)  # need train to be %=0 by batch_size
x_test = x_data[:split_point]
y_test = y_data[:split_point]
x_train = x_data[split_point:]
y_train = y_data[split_point:]

print("ct of y_train = gesture is {:d}/{:d} and ct of y_test = gesture is {:d}/{:d}"
      .format(sum([1 if cat[1] == 1 else 0 for cat in y_train]),
              len(y_train),
              sum([1 if cat[1] == 1 else 0 for cat in y_test]),
              len(y_test)))


input_shape = (frame_size, number_data_columns, 1)
model = Sequential()
model.add(Conv2D(2, (2, 2), activation='relu',
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(2, (2, 2), activation='relu',
                 padding='same',
                 input_shape=input_shape))
model.add(MaxPool2D((2,2)))
# model.add(LSTM(units=lstm_units,
#                activation='relu',
#                recurrent_activation='hard_sigmoid',
#                stateful=False,
#                batch_size=batch_size))
# model.add(Dense(units=16))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          class_weight=class_weights,
          validation_data=(x_test, y_test))

model.save('last_model.h5')

predictions = model.predict(x_data)

print("done")

# TODO: How to make validation greatly penalize false positives and not care about false negatives?
# TODO: What does LSTM units do? How does the temporal aspect get carried along?
# TODO: Still, how do you quantify: A gesture happened in this window sometime?
# TODO: Also, what should I record as a gesture in order to make it easier for training?

