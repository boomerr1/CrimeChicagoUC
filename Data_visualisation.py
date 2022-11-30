import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt


# path = 'data/shapefile/geo_export.shp'
# chicago = gpd.read_file(path)
# # chicago = africa[africa['ISO3'].notna()]
# chicago.plot()
# plt.title("Chicago")
# plt.xticks([])
# plt.yticks([])
# plt.show()


# Load data
print('Loading data...')
crime_df = pd.read_csv('data/Crimes_-_2001_to_2021.csv')
print('Loaded the data!')

# Cleaning dataframe
crime_df.drop(columns=['Case Number','Block','IUCR','Primary Type', # IUCR of Primary Type ?
                        'Description','Location Description','Arrest', # Location Description voor LSTM ?
                        'Domestic','Beat','District','Ward','Community Area', # Community Are voor visualisatie
                        'FBI Code','Updated On','Location'], inplace=True)
crime_df = crime_df.rename(columns={'X Coordinate':'x', 'Y Coordinate':'y'})
crime_df.dropna(inplace=True)
print(crime_df.head(3))
print(crime_df.shape)

# Extract last month of data
print('Converting to datetime column')
month_crime_df = crime_df.loc[crime_df['Year'] == 2021]
month_crime_df['Date'] = pd.to_datetime(month_crime_df['Date'])
month_crime_df = month_crime_df.loc[crime_df['Date'] > '11/31/2021']
print(month_crime_df.shape)

crs="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

# Dataframe to GeoDataframe
gdf = gpd.GeoDataFrame(month_crime_df, geometry=gpd.points_from_xy(month_crime_df.x, month_crime_df.y),
    crs=crs)

# gdf.plot(markersize=.1, figsize=(8, 8))
# plt.show()

# Plot crime locations
# ax = gdf.plot(markersize=.1, figsize=(12, 8))
# plt.autoscale(False)
# path = 'data/shapefile/geo_export.shp'
# chicago = gpd.read_file(path)
# chicago.plot(ax=ax, edgecolor='black')
# plt.title("Chicago")
# ax.axis('off')
# plt.show()


xmin, ymin, xmax, ymax = gdf.total_bounds
n_cells=30
cell_size = (xmax-xmin)/n_cells
grid_cells = []
for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
    for y0 in np.arange(ymin, ymax+cell_size, cell_size):
        x1 = x0-cell_size
        y1 = y0+cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1)  )
cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], 
                                 crs=crs)

ax = gdf.plot(markersize=.1, figsize=(12, 8))
plt.autoscale(False)
cell.plot(ax=ax, facecolor="none", edgecolor='grey')
ax.axis("off")

merged = gpd.sjoin(gdf, cell, how='left', op='within')

merged['n_fires']=1
dissolve = merged.dissolve(by="index_right", aggfunc="count")
cell.loc[dissolve.index, 'n_fires'] = dissolve.n_fires.values

path = 'data/shapefile/geo_export.shp'
ax = cell.plot(column='n_fires', figsize=(12, 8), cmap='viridis', vmax=5000, edgecolor="grey")
plt.autoscale(False)
world = gpd.read_file(path)
world.to_crs(cell.crs).plot(ax=ax, color='none', edgecolor='black')
ax.axis('off')