import tensorflow as tf
from tensorflow.keras import Sequential, Input, layers, Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import timeseries_dataset_from_array
import geopandas as gpd
import numpy as np
import shapely


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

# Data laden
X_crimes_only = np.load('../data/train_data.npy')
# X_crimes_only = X_crimes_only.reshape(X_crimes_only.shape[0], X_crimes_only.shape[1], X_crimes_only.shape[2], 1)
# print(X_crimes_only.shape)

lookback = 7
batch_size = 32

data = X_crimes_only[:-lookback]
targets = X_crimes_only[lookback:]

train_gen = TimeseriesGenerator(
    X_crimes_only,
    X_crimes_only,
    length=lookback,
    batch_size=batch_size,
    shuffle=False
)

# for i in range(len(train_gen)):
#     x, y = train_gen[i]
#     break

# print(x[1][6].shape, y[0].shape)
# print(np.testing.assert_array_equal(x[1][6], y[0])) # True
# print(np.testing.assert_array_equal(x[1][6], X_crimes_only[7])) # True
# print(np.testing.assert_array_equal(x[2][6], y[1])) # True

# # y not dit


# train_gen = timeseries_dataset_from_array(
#     data=data,
#     targets=targets,
#     sequence_length=lookback
# )

# training sequence: [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# 1 training sequence
# X_1 = [1,2,3,4,5,6,7]
# X_2 = [2,3,4,5,6,7,8]
# ...
# X_7 = [7,8,9,10,11,12,13]

# X_1_targets = [2,3,4,5,6,7,8]
# X_2_targets = [3,4,5,6,7,8,9]
# ...
# X_7_targets = [8,9,10,11,12,13,14]


# of X_8[8,9,10,11,12,13,14]
# of 14


# Dit is ook een mogelijke interpretatie van de paper
inputs = Input(shape=(lookback, 55, 50, 1))
mask = Input(shape=(mask.shape[0], mask.shape[1], 1))
convlstm1 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
bathnorm1 = layers.BatchNormalization()(convlstm1)
convlstm2 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(bathnorm1)
# batchnorm2 = layers.BatchNormalization()(convlstm2)
convlstm3 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(inputs)
batchnorm3 = layers.BatchNormalization()(convlstm3)
convlstm4 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh', return_sequences=True)(convlstm3)
concatenation = layers.concatenate([convlstm2, convlstm4])
conv2d = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same", activation='tanh')(concatenation)
outputs = layers.Multiply()([conv2d, mask])

model = Model(inputs=[inputs,mask], outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=10)

