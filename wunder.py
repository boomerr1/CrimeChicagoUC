import arrow # learn more: https://python.org/pypi/arrow
from WunderWeather import weather # learn more: https://python.org/pypi/WunderWeather

api_key = 'a6f06ba0a07042c1b06ba0a07072c1c9'

location = 'IL/Chicago'
extractor = weather.Extract(api_key)

print(extractor.alerts)

# alerts
# response = extractor.alerts(location)

# # astronomy
# response = extractor.astronomy(location)

# # geolookup
# response = extractor.geolookup(location)

# # history
# date = arrow.get("20170601","YYYYMMDD")
# response = extractor.date(location,date.format('YYYYMMDD'))

# # addl date detail
# for observation in response.observations:
#     print("Date:",observation.date_pretty)
#     print("Temp:",observation.temp_f)