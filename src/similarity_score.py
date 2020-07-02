import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = None
data = None


def get_cleaned_price(raw_price):
    m = re.search(r"[-+]?\d*\.?\d+|\d+", raw_price)
    group = m.group()
    return float(group) if 'Lakh' in raw_price else float(group) * 100


def get_label_for_price(price):
    if price < 5:
        return 'EconomicalRange'
    elif 5 < price < 8:
        return 'MidRange'
    elif 8 < price < 14:
        return 'HighRange'
    elif 14 < price < 22:
        return 'VeryHighRange'
    elif 22 < price < 40:
        return 'Luxury'
    elif 40 < price < 80:
        return 'HighLuxury'
    return 'UltimateLuxury'


def get_label_for_engine(engine):
    if engine == -1:
        return 'NA'
    elif 0 < engine < 1000:
        return 'SmallEngine'
    elif 1000 < engine < 1400:
        return 'MediumEngine'
    elif 1400 < engine < 2000:
        return 'BigEngine'
    elif 2000 < engine < 4000:
        return 'VeryBigEngine'
    return 'ExtremelyBigEngine'


def get_recommendations(idx):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    car_indices = [score[0] for score in sim_scores]
    similarity = [score[1] for score in sim_scores]
    # recommendation = pd.DataFrame({'model': data['model_name'].iloc[car_indices], 'similarity':similarity})
    recommendation = pd.DataFrame({'vid': data['vid'].iloc[car_indices], 'similarity':similarity})
    recommendation = recommendation[(recommendation['similarity'] > .74) & (recommendation['vid'] != idx)]
    return recommendation['vid']


def compute_similarities():
    global data
    data = pd.read_csv('data/cars_data.csv')
    data["engine"].fillna(-1, inplace=True)
    data = data.dropna()
    data['starting_price'] = data['min_price'].apply(lambda raw_price: get_cleaned_price(raw_price))
    data['segment'] = data['starting_price'].apply(lambda price: get_label_for_price(price))
    data['engine_category'] = data['engine'].apply(lambda engine: get_label_for_engine(engine))
    data['soup'] = data['body_type'] + ' ' + data['segment'] + ' ' + data['engine_category']
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data['soup'])
    global cosine_sim
    cosine_sim = cosine_similarity(count_matrix, count_matrix)


# cosine_sim, data = get_similarities()
# print(get_recommendations(221, cosine_sim, data))

