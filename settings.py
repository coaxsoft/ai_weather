"""Keep all application global settings here
"""
import os

PYTHON_INTERPRETER = 'python'

user = os.environ.get("MONGO_INITDB_ROOT_USERNAME")
password = os.environ.get("MONGO_INITDB_ROOT_PASSWORD")

MONGO_URI = f'mongodb://{user}:{password}@127.0.0.1:27017'
MONGO_DB = os.environ.get('MONGO_DB', 'weather_db')

_predict_services = (
    'gismeteo',
    'meteoprog',
    'weather',
    'weatherforecast',
    'yahoo',
)
_actual_service = ('actual_weather',)
MONGO_COLLECTIONS = {
    'predict_services': _predict_services,
    'actual_service': _actual_service[0],
    'services': _predict_services + _actual_service,
    'weights': 'weights',
    'errors': 'errors',
    'produced_data': 'produced_data',
    'classes': 'weather_classes',
    'cities': 'cities',
    'about_us': 'about'
}

# Weather params
DEFAULT_WEATHER_MEASUREMENT_UNITS = {
    'temperature': '°C',
    'feels_temperature': '°C',
    'humidity': '%',
    'precipitation': '%',
    'pressure': 'mmHg',
    'wind_direction': 'normal',
    'wind_speed': 'm/s'
}

HARD_LIMIT = 250

LENGTH_FORECAST = 7
