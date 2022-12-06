import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt
from matplotlib import cm

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
            mask[i,j] = np.nan

train_data = np.load('../data/train_data.npy')
test_data = np.load('../data/test_data.npy')

historical_average = np.mean(train_data, axis=0)

historical_average *= mask
plt.imshow(historical_average, vmax=2, cmap='jet')
plt.axis('off')
plt.show()

mse = np.nanmean(np.square(np.subtract(test_data, np.repeat([historical_average], len(test_data), axis=0))))
print('MSE: ', mse.mean())
print('RMSE: ', np.sqrt(mse).mean())