# from tensorflow.keras import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import geopandas as gpd
import numpy as np
import shapely

path = 'data/shapefile/geo_export.shp'
chicago = gpd.read_file(path)
chicago = chicago.dissolve()

xmin, ymin, xmax, ymax = chicago.total_bounds
n_x_cells = 100
x_cell_size = (xmax - xmin) / n_x_cells
y_cell_size = round(((ymax - ymin)/(xmax - xmin))*n_x_cells)
mask = []
x_arange = np.arange(xmin, xmax+x_cell_size, x_cell_size)
y_arange = np.arange(ymin, ymax+y_cell_size, y_cell_size)
for xi, x0 in zip(range(len(x_arange)), x_arange):
    for yi, y0 in zip(range(len(y_arange)), y_arange):
        x1 = x0-x_cell_size
        y1 = y0+y_cell_size
        box = shapely.geometry.box(x0, y0, x1, y1)
        if not chicago.intersection(box).any():
            mask.append((xi,yi))
print(mask)

# lookback = 7
# batch_size = 32

# train_gen = TimeseriesGenerator(
#     data,
#     targets,
#     length=lookback,
#     sampling_rate=1,
#     stride=1,
#     start_index=0,
#     end_index=None,
#     shuffle=False,
#     reverse=False,
#     batch_size=batch_size
# )


# model = Sequential()
# model.add(layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(None, 1, 1, 1)))
# model.add.Conv1D(32, 5, activation='relu')
# model.add.Multiply(mask)

# model.compile(optimizer='adam', loss='cross-entropy', metrics=['accuracy'])
