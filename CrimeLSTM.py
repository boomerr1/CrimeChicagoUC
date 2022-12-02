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
mask = np.ones((n_x_cells, n_x_cells))
grid = {"mask":[], "geometry":[]}
for i in range(mask.shape[0]):
    x0 = i * x_cell_size + xmin
    x1 = x0 + x_cell_size
    for j in range(mask.shape[1]):
        y0 = j * y_cell_size + ymin
        y1 = y0 + y_cell_size
        box = shapely.geometry.box(x0, y0, x1, y1)
        if not chicago.intersection(box).any():
            mask[-j-1,i] = 0
            grid["mask"].append(0)
        else:
            grid["mask"].append(1)
        grid["geometry"].append(box)

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
