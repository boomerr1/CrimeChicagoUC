import tensorflow as tf
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt

path = '../data/shapefile/geo_export.shp'
chicago = gpd.read_file(path)
chicago = chicago.dissolve()

xmin, ymin, xmax, ymax = chicago.total_bounds
n_x_cells = 50
x_cell_size = (xmax - xmin) / n_x_cells
n_y_cells = round(((xmax - xmin)/(ymax - ymin))*n_x_cells)
y_cell_size = (ymax - ymin) / n_y_cells
mask = np.ones((n_y_cells, n_x_cells))
x_arange = np.arange(xmin, xmax+x_cell_size, x_cell_size)
y_arange = np.arange(ymin, ymax+y_cell_size, y_cell_size)
for i, y0 in zip(range(n_y_cells-1, -1, -1), y_arange):
    for j, x0 in zip(range(n_x_cells), x_arange):
        x1 = x0-x_cell_size
        y1 = y0+y_cell_size
        box = shapely.geometry.box(x0, y0, x1, y1)
        if not chicago.intersection(box).any():
            mask[i,j] = 0
mask = tf.keras.backend.constant(mask)
mask = tf.expand_dims(mask, -1)

lookback = 7
batch_size = 4

train_X_crimes_only = np.load('../data/train_data.npy')
test_X_crimes_only = np.load('../data/test_data.npy')

train_X_crimes_only = tf.expand_dims(train_X_crimes_only, -1)
test_X_crimes_only = tf.expand_dims(test_X_crimes_only, -1)

train_gen = TimeseriesGenerator(
    train_X_crimes_only,
    train_X_crimes_only,
    length=lookback,
    batch_size=batch_size,
    shuffle=False
)
test_gen = TimeseriesGenerator(
    test_X_crimes_only,
    test_X_crimes_only,
    length=lookback,
    batch_size=batch_size,
    shuffle=False
)

def masked_MSE_loss(y_true, y_pred):
    y_pred_masked = tf.math.multiply(y_pred, mask)
    mse = tf.keras.losses.mean_squared_error(y_true = y_true, y_pred = y_pred_masked)
    return mse

# Define the input tensors
inputs = Input(shape=(lookback, *train_X_crimes_only.shape[1:]))

# First stack of convlstm layers
convlstm1 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
bathnorm1 = layers.BatchNormalization()(convlstm1)
convlstm2 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=False)(bathnorm1)

# Second stack of convlstm layers
convlstm3 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
batchnorm2 = layers.BatchNormalization()(convlstm3)
convlstm4 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=False)(batchnorm2)

# Concatenate outputs of two stacks
concatenation = layers.concatenate([convlstm2, convlstm4])
outputs = layers.Conv2D(filters=1, kernel_size=1, padding="same", activation='tanh')(concatenation)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss=masked_MSE_loss, metrics=['mae'])

# Train the model
model.fit(train_gen, epochs=1)


# Create test prediction
test_pred = model.predict(test_gen)
test_pred *= mask

np.save('../data/final_pred_ConvLSTM_Cr.npy', test_pred)

test_X_crimes_only[-1][mask == False] = np.nan

last_day_truth = test_X_crimes_only[-1]
last_day_pred = test_pred[-1]
last_day_pred[mask == False] = np.nan

fig, axs = plt.subplots(1, 2)
fig.set_figheight(7)
fig.set_figwidth(14)
axs[0].imshow(last_day_truth)
axs[1].imshow(last_day_pred)
plt.show()

ground_truth = test_X_crimes_only[7:]
mse = np.nanmean(np.square(np.subtract(ground_truth, test_pred)))
print(f'Hetero-ConvLSTM - MSE: {mse.mean():.3f}')
print(f'Hetero-ConvLSTM - RMSE: {np.sqrt(mse).mean():.3f}')