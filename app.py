from flask import Flask, render_template, request, jsonify

from src.data_provider import get_cars, get_all_cars, get_car
from src.similarity_score import compute_similarities, get_recommendations


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('carlist.html', cars=get_all_cars())


@app.route('/car-recommondar-system/<int:carid>')
def recommendation(carid):
    similar_car_ids = get_recommendations(carid)
    similar_cars = get_cars(similar_car_ids)
    return render_template('recommender.html', car=get_car(carid), similarcars=similar_cars)


if __name__ == "__main__":
    compute_similarities()
    app.run(debug=True)
