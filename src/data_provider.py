import pandas as pd


def get_all_cars():
    df = pd.read_csv('data/cars_data.csv')
    dicts = df.to_dict('records')
    return dicts


def get_car(carid):
    df = pd.read_csv('data/cars_data.csv')
    df = df[df['vid'] == carid]
    dicts = df.to_dict('records')
    return dicts[0]


def get_cars(carids):
    df = pd.read_csv('data/cars_data.csv')
    df = df[df['vid'].isin(carids)]
    dicts = df.to_dict('records')
    return dicts
