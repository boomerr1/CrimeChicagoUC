# CrimePredictionChicago
A machine learning model to predict criminal activity in the city Chicago.

## Data:

[Chicago crime data](https://data.cityofchicago.org/Public-Safety/Crimes-One-year-prior-to-present/x2n5-8w5q/data)

[Light intensity map](https://www.nasa.gov/sites/default/files/thumbnails/image/26247384716_9281df96cc_o.jpg)

[Street lights](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Street-Lights-One-Out-No-Dupl/idsv-mf2w)

[Alley lights](https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Alley-Lights-Out-No-Duplicate/up7z-t43p)

[Weather data](https://www.visualcrossing.com/weather/weather-data-services)

[Shapefile](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Neighborhoods/bbvz-uum9)

[Police Station Locations](https://data.cityofchicago.org/Public-Safety/Police-Stations/z8bn-74gv)

[Crime severity](https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/datasets/crimeseverityscoreexperimentalstatistics)

[Poverty Indicators](https://data.cityofchicago.org/Health-Human-Services/Poverty-Indicators-by-COmmunity-Area/c44j-fgcy/data)

# Roadmap:
- [x] Recreate the model for crime prediction in Chicago
- [x] Add and preprocess only a subset (most important) data
    - [x] create mask to delete predictions outside of Chicago.
    - [x] Use the last 10 years (2011-2021)
- [x] Evaluate the model: MSE & RMSE
    - [x] Baseline historical average
    - [x] LSTM
- [ ] Add more data that might improve accuracy
    - [ ] street and alley lights
    - [ ] poverty
    - [x] weather
      - [x] convert to (ndays,55,50) sized array
    - [x] severity of crimes
      - [x] column of severity to be used as weight for the average

# Log
30-11-2022 21:11:
In photoshop de raw sattelite image bewerken en alignen met google maps screenshot:
    - perspective warp 
    - rotatie
    - breedte kleiner met stretch
    - lengte iets kleiner met stretch.
Mislukt so far, word op zijspoor gezet

5-12-2022 13:00
LSTM inputs proberen goed te krijgen als time series met de juiste shape en size.

5-12-2022 20:04
train set:  2011 - 2020
test set:   2021
Baseline: historical average elke dag in het hele test set jaar hetzelfde.