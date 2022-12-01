from shapely.geometry import Polygon
import numpy as np
from shapely.prepared import prep
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


def grid_bounds(geom, nx, ny):
    minx, miny, maxx, maxy = geom.bounds
    gx, gy = np.linspace(minx,maxx,nx+1), np.linspace(miny,maxy,ny+1)
    grid = []
    for i in range(nx):
        for j in range(ny):
            poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
            grid.append( poly_ij )
    return grid


def partition(geom, nx, ny):
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, nx, ny)))
    return grid


path = 'data\shapefile\geo_export.shp'
chiraq = gpd.read_file(path)
chiraq = chiraq[chiraq['pri_neigh'] != "O'Hare"]
geom = chiraq.dissolve()['geometry'].values[0]
print(geom.bounds)

# grid = partition(geom, 5, 3)

# fig, ax = plt.subplots(figsize=(8, 8))
# chiraq.boundary.plot(ax=ax, edgecolor='black')
# gpd.GeoSeries(grid).boundary.plot(ax=ax, edgecolor='red')
# plt.title('Chicago Grid for sliding window (2x2) LSTM')
# # hide ticks
# plt.xticks([]),plt.yticks([])

im_path = "chicago_night_processed.png"

im = Image.open(im_path)

plt.imshow(im, alpha=0.5)
plt.show()
