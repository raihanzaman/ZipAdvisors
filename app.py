from flask import Flask, request, render_template
from ml_backend import prediction_dropdowns, plot_kalshi_data, plot_polymarket_data
from flask_cors import CORS
from datetime import datetime
import os
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/', methods = ['GET', 'POST'])
def index():
    predictions = prediction_dropdowns()
    if request.method == 'GET':
        return render_template('index.html', predictions=predictions)
    if request.method == 'POST':
        if 'submit-kalshi' in request.form:
            kalshi = request.form
            kalshi_url = plot_kalshi_data(kalshi['select-market'])
            return render_template('index.html', predictions=predictions, kalshi_url=kalshi_url)
        if 'submit-poly' in request.form:
            polymarket = request.form
            polymarket_url = plot_kalshi_data(polymarket['select-market'])
            return render_template('index.html', predictions=predictions, polymarket_url=polymarket_url)
'''
@app.route('/api/players/<market>')
def get_players(market):
    engine = POLYMARKET_ENGINE if market.lower() == 'polymarket' else KALSHI_ENGINE
    players = get_player_list(engine)
    return jsonify({"players": players})

@app.route('/api/market_data/<market>', methods=['GET'])
def get_market_data_route(market):
    player = request.args.get('player')
    yes_no = request.args.get('yes_no', 'yes')
    
    if not player:
        return jsonify({"error": "Player parameter is required"}), 400
    
    engine = POLYMARKET_ENGINE if market.lower() == 'polymarket' else KALSHI_ENGINE
    data = get_market_data(engine, player, yes_no)
    return jsonify(data)
'''
if __name__ == '__main__':
    app.run(debug=True, port=5000)