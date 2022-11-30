# TODO: 
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# TODO: 
path = 'data\Boundaries - Neighborhoods\geo_export_7ad36bff-e877-4687-85bf-007c957c1dd9.shp'
africa = gpd.read_file(path)
# africa = africa[africa['ISO3'].notna()]
africa.plot()
plt.title("Africa 2020")
plt.xticks([])
plt.yticks([])
plt.show()

# TODO: 

# TODO: 