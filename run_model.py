"""This is the main script that is embedded with Mongo database
and and can be run to reduce new weights, produce new data and
get errors.

**************
Usage examples
**************

The script is build like CLI application, so you can run it simply
like::

    python __main__.py REDUCE --city Kalush --country Afghanistan -m 7 -l 30

You may already figure out command line arguments:

* :param positional: first argument is positional and necessary.
                      It can take values: ``REDUCE|PRODUCE``. Use
                      ``REDUCE`` to reduce weights and write them
                      to Mongo; use ``PRODUCE`` to calculate new
                      weather for the latest data in DB limited
                      by ``--limit``
* :param --city: city name
* :param --country: country name
* :param -m --max: max forecast distance for which to reduce
                  weights, i. e. for 0:max. Default is 0
* :param -l --limit: limit the amount of values that are used in
                    weights reducing. Default 7

By default the script saves data to local instance of MongoDB.
Default database name is ``weather_db``. It writes data (by
default) to the next collections:

* Weights to ``weights`` collection;
* Errors to ``errors`` collection;
* Produced data to ``produced_data`` collection.

By default the script uses Euclidian metrics to calculate weights
and errors.
"""
import argparse
import time

import pymongo

import settings
from settings import MONGO_URI, MONGO_DB
from prediction_model import to_list
from prediction_model.converters import MongoIntersectConverter, MongoUnionConverter
from prediction_model.estimators import PostProcessEstimator
from prediction_model.io.readers import MongoReader
from prediction_model.io.writers import MongoWriter
import prediction_model.estimators as estimators
from prediction_model.postprocessors import MaxWeightPostprocessor
from prediction_model.preprocessors import Word2ClassPreprocessor


COLLECTION = 'gismeteo'
CHECK_WORDS_COLLECTION = 'weather_classes_calculate'


def parse_args():
    """Function that parse command line arguments.
    It is just for encapsulation
    :return command line arguments
    """
    argp = argparse.ArgumentParser()
    argp.add_argument('script', help='What script to run: REDUCE or PRODUCE. Reduce '
                                     'calculates new weights and save them into Mongo '
                                     'while Produce calculates new output based on the '
                                     'last calculated weights')
    argp.add_argument('--city', dest='city', type=str, default=None, help='Name of city')
    argp.add_argument('--country', dest='country', type=str, default=None, help='Name of country')
    argp.add_argument('-m', '--max', dest='max', default=0, type=int,
                      help='Max value of forecast distance')
    argp.add_argument('-l', '--limit', dest='limit', default=30, type=int,
                      help='Limit the range of the latest data on which model would be learnt')
    return argp.parse_args()


def retrieve_check_words(collection):
    """Retrieves from db check words of weather description
    :param collection: name of check words collection
    :return dictionary of check words
    """
    mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]
    res = mongo[collection].find({}, {'_id': 0}).sort([('$natural', -1)]).limit(1)
    return {str(k): v for k, v in res[0]['data'].items()}


def retrieve_max_forecast(collection, distance=None):
    """Sometimes when user pass distance parameter too high
    it is needed to settle him down a little
    :param collection: name of weather collection
    :param distance: forecast distance
    :return max forecast distance
    """
    mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]
    res = mongo[collection].aggregate([{'$group': {'_id': 0,
                                                   'max': {'$max': '$forecast_distance'}}}])
    res = next(res)['max']
    if distance is not None:
        if distance < res:
            return distance
    return res


def produce(city: str, country: str, distance: int, limit: int, key_map: dict):
    """Main function that produce new weather
    results calculating them from weights already
    existing in the MongoDB
    :param city: city name
    :param country: country name
    :param distance: forecast distance
    :param limit: count of days to limit producing
    :param key_map: dictionary of weather parameters
    """
    print('Begin producing')
    reader = MongoReader(MONGO_URI, MONGO_DB)
    spent_time = 0
    distance = retrieve_max_forecast(COLLECTION, distance)
    writer = MongoWriter(MONGO_URI, MONGO_DB)
    for d in range(distance + 1):
        prev_time = time.time()
        converter = MongoIntersectConverter.from_mongo(mongo_uri=MONGO_URI,
                                                       mongo_db=MONGO_DB,
                                                       data_collections=settings.MONGO_COLLECTIONS[
                                                           'predict_services'],
                                                       real_collection=settings.MONGO_COLLECTIONS['actual_service'],
                                                       city=city,
                                                       country=country,
                                                       distance=d,
                                                       limit=limit)
        produce_converter = MongoUnionConverter.from_mongo(mongo_uri=MONGO_URI,
                                                           mongo_db=MONGO_DB,
                                                           data_collections=settings.MONGO_COLLECTIONS[
                                                               'predict_services'],
                                                           city=city,
                                                           country=country,
                                                           distance=d,
                                                           limit=limit)
        try:
            conv = converter.convert(key_map)
            data, _ = produce_converter.convert(key_map)

            # 0 means that we search weights with 0 forecast distance. Actually,
            # this isn't totally right because we lose some ability of model
            # to calculate weights depending on "farness" from current date to
            # prediction date. But it works! and is acceptable because in any
            # way we get only approximate solution
            reader.search(city, country, 0)
            res = estimators.produce(reader, data)
            cv_res = estimators.produce(reader, conv[0])

            cv_error = estimators.cv(cv_res, conv[1])
            cv_error['data'] = to_list(cv_error['data'])

            res['data'] = to_list(res['data'])

            # 0 means that we write data with forecast distance always 0.
            # actually, you can read above comment for that.
            writer.supplement({
                'city': city,
                'country': country,
                'forecast_distance': 0
            })
            writer.write_produced(res)
            writer.write_errors(cv_error)
        except TypeError as error:
            print(f"\nTYPE_ERROR: {city}, {country}:{d}, {error}\n")
        except IndexError as error:
            print(f"\nINDEX_ERROR: {city}, {country}:{d}, {error}\n")
        except ValueError as error:
            print(f"\nVALUE_ERROR: {city}, {country}:{d}, {error}\n")
        spent_time += time.time() - prev_time
        print(f"{city}, {country}:{d} was processed; {time.time() - prev_time: .3f}/{spent_time: .3f}")


def reduce(city: str, country: str, distance: int, limit: int, key_map: dict):
    """Main function that calculates weights of services"""
    print('Begin reducing')

    spent_time = 0
    distance = retrieve_max_forecast(COLLECTION, distance)
    writer = MongoWriter(MONGO_URI, MONGO_DB)
    for d in range(distance + 1):
        prev_time = time.time()
        converter = MongoUnionConverter.from_mongo(mongo_uri=MONGO_URI,
                                                   mongo_db=MONGO_DB,
                                                   data_collections=settings.MONGO_COLLECTIONS['predict_services'],
                                                   real_collection=settings.MONGO_COLLECTIONS['actual_service'],
                                                   city=city,
                                                   country=country,
                                                   distance=d,
                                                   limit=limit)
        try:
            est = PostProcessEstimator.from_converter(converter, key_map)
            est.reduce(weights_postprocessors={'class': (MaxWeightPostprocessor(),)})
            writer.supplement({
                'city': city,
                'country': country,
                'forecast_distance': d
            })
            writer.write_weights(est)
        except ValueError as error:
            print(f"\nVALUE_ERROR: {city}, {country}:{d}, {error}\n")
        except IndexError as error:
            print(f"\nINDEX_ERROR: {city}, {country}:{d}, {error}\n")
        spent_time += time.time() - prev_time
        print(f"{city}, {country}:{d} was processed; {time.time() - prev_time: .3f}/{spent_time: .3f}")


if __name__ == '__main__':
    args = parse_args()

    key_map_object = {
        'temperature_max': (
            ('temperature', 'max'),
        ),
        'temperature_min': (
            ('temperature', 'min'),
        ),
        'feels_temperature_max': (
            ('feels_temperature', 'max'),
        ),
        'feels_temperature_min': (
            ('feels_temperature', 'min'),
        ),
        'pressure_max': (
            ('pressure', 'max'),
        ),
        'pressure_min': (
            ('pressure', 'min'),
        ),
        'humidity_max': (
            ('humidity', 'max'),
        ),
        'humidity_min': (
            ('humidity', 'min'),
        ),
        'precipitation_max': (
            ('precipitation', 'max'),
        ),
        'precipitation_min': (
            ('precipitation', 'min'),
        ),
        'wind_direction_max': (
            ('wind_direction', 'max'),
        ),
        'wind_direction_min': (
            ('wind_direction', 'min'),
        ),
        'wind_speed_max': (
            ('wind_speed', 'max'),
        ),
        'wind_speed_min': (
            ('wind_speed', 'min'),
        ),
        'class': (
            ('description',),
            (Word2ClassPreprocessor(retrieve_check_words(CHECK_WORDS_COLLECTION)),)
        )
    }

    if not args.city:
        print('City name is empty. Please, Honorable Sir, provide it or pass to command line argument --city')
    elif not args.country:
        print('Country name is empty. Please, Honorable Sir, provide it or pass to command line argument --country')
    else:
        if args.script.upper() == 'REDUCE':
            reduce(args.city, args.country, args.max, args.limit, key_map_object)
        elif args.script.upper() == 'PRODUCE':
            produce(args.city, args.country, args.max, args.limit, key_map_object)
        else:
            print('You called nothing :(')
