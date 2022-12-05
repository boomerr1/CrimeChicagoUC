import tensorflow as tf
from tensorflow.keras import Sequential, Input, layers, Model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import geopandas as gpd
import numpy as np
import shapely


path = 'data/shapefile/geo_export.shp'
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
X_crimes_only = np.load('data/X_crimes_only.npy')
# X_crimes_only = X_crimes_only.reshape(X_crimes_only.shape[0], X_crimes_only.shape[1], X_crimes_only.shape[2], 1)
print(X_crimes_only.shape)

t = 0
for day in X_crimes_only:
    t+=1
print(t)

lookback = 7
batch_size = 32

train_gen = TimeseriesGenerator(
    X_crimes_only,
    X_crimes_only,
    length=lookback,
    batch_size=batch_size
)

# Create ConvLSTM model that predicts the image of the next day with logits and labels of the same shape
model = Sequential()
model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(lookback, 55, 50, 1), padding='same', return_sequences=True))
model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True))
model.add(layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(train_gen, epochs=10)

# inputs = Input(shape=(lookback, 55, 50, 1))
# mask = Input(shape=(mask.shape[0], mask.shape[1], 1))
# convlstm1 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh')(inputs)
# bathnorm1 = layers.BatchNormalization()(convlstm1)
# convlstm2 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh')(bathnorm1)
# batchnorm2 = layers.BatchNormalization()(convlstm2)
# convlstm3 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh')(batchnorm2)
# batchnorm3 = layers.BatchNormalization()(convlstm3)
# convlstm4 = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', activation='tanh')(batchnorm3)
# conv2d = layers.Conv2D(filters=1, kernel_size=(1,1), padding="same", activation='tanh')(convlstm4)
# outputs = layers.Multiply()([conv2d, mask])

# model = Model(inputs=[inputs,mask], outputs=outputs)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_gen)

