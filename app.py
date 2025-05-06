from flask import Flask, request, send_file, render_template
from ml_backend import plot_kalshi_data, plot_polymarket_data
from flask_cors import CORS
from datetime import datetime
import os
import re
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        #kalshi = request.form['kalshi-market']
        #polymarket = request.form['polymarket-market']
        img_io = plot_polymarket_data()
        img_io2 = plot_kalshi_data()
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
        img2_base64 = base64.b64encode(img_io2.getvalue()).decode('ascii')
        poly_url = f'data:image/png;base64,{img_base64}'
        kalshi_url = f'data:image/png;base64,{img2_base64}'
        return render_template('index.html', polymarket_url=poly_url, kalshi_url=kalshi_url)
        '''
        if 'submit-poly' in request.form:
            img_io = plot_polymarket_data()
            img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
            plot_url = f'data:image/png;base64,{img_base64}'
            return render_template('index.html', polymarket_url=plot_url)
        if 'submit-kalshi' in request.form:
            img_io = plot_kalshi_data()
            img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
            plot_url = f'data:image/png;base64,{img_base64}'
            return render_template('index.html', kalshi_url=plot_url)
        '''
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