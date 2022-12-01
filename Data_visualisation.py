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



# Dataframe to GeoDataframe
crs="EPSG:4326"
gdf = gpd.GeoDataFrame(month_crime_df, geometry=gpd.points_from_xy(month_crime_df.Longitude, month_crime_df.Latitude),
    crs=crs)
print(gdf.head())

xmin, ymin, xmax, ymax = gdf.total_bounds
n_cells=50
cell_size = (xmax-xmin)/n_cells
grid_cells = []
for x0 in np.arange(xmin, xmax+cell_size, cell_size):
    for y0 in np.arange(ymin, ymax+cell_size, cell_size):
        x1 = x0-cell_size
        y1 = y0+cell_size
        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

merged = gpd.sjoin(gdf, cell, how='left', predicate='within').dropna()
print(merged.head())
print(merged.shape)
merged['n_crimes'] = 0
dissolve = merged.dissolve(by="index_right", aggfunc="count")
print(dissolve.head())
cell.loc[dissolve.index, 'n_crimes'] = dissolve.n_crimes.values
print(cell.head())

path = 'data/shapefile/geo_export.shp'
print(cell.head())
ax = cell.plot(column='n_crimes', figsize=(12, 8), cmap='jet', vmax=5000, edgecolor="grey")
chicago = gpd.read_file(path)
chicago.to_crs(cell.crs).plot(ax=ax, color='none', edgecolor='black')
ax.axis('off')

gdf.plot(ax=ax, markersize=.1, figsize=(12, 8))

plt.show()