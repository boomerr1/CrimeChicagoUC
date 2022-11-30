# TODO: 
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# TODO: 
path = 'data\shapefile\geo_export.shp'
africa = gpd.read_file(path)
# africa = africa[africa['ISO3'].notna()]
africa.plot()
plt.title("Chicago")
plt.xticks([])
plt.yticks([])
plt.show()

# TODO: 

# TODO: 