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
    grid = prep(geom)
    # grid = list(filter(grid.intersects, grid_bounds(geom, nx, ny)))
    return grid_bounds(geom, nx, ny)

crs = "EPSG:4326"
path = 'data\shapefile\geo_export.shp'
chiraq = gpd.read_file(path)
# chiraq = chiraq[chiraq['pri_neigh'] != "O'Hare"]
geom = chiraq.dissolve()['geometry'].values[0]

grid = partition(geom, 55//15, 50//15)

fig, ax = plt.subplots(figsize=(8, 8))
chiraq.plot(ax=ax, edgecolor='black')
gpd.GeoSeries(grid, crs=crs).boundary.plot(ax=ax, edgecolor='red')
plt.title("Windows for Conv-LSTM's")
# hide ticks
plt.xticks([]),plt.yticks([])
plt.show()
