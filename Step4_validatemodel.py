import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras.models import Model, load_model
import numpy as np

models: [Model] = []
model_names = ["./models/A_best_model.h5", "./models/B_good_model.h5", "./models/C_last_model.h5"]

for name in model_names:
    model = load_model(name)
    models.append(model)

batch_size = 32
num_classes = 2

# Data where col 0 is the data picture, and col 1 is the label
data_column = 0
gesture_classification_column = 1
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

x_data = np.stack(df[:, data_column])
y_data = df[:, gesture_classification_column]

predictions = []
for model in models:
    prediction = model.predict(x_data, batch_size=batch_size)
    predictions.append(prediction)

predictions = np.array(predictions)[:, 0:2000, :]

for i in range(predictions.shape[1]):
    prediction = predictions[:, i, :]
    print("{:d}: {:s} for truth: y={:d}"
          .format(i, str(prediction).replace("\n", "-"), y_data[i]))

