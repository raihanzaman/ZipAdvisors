from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from standardize_names import standardizeColumnNames
from datetime import datetime
import os
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests
'''
# Configuration
EVENT_ID = "masters_2025"  # Replace with your event_id
MYSQL_USER = os.getenv("MYSQL_USER", "your_username")  # Replace with your MySQL username
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "your_password")  # Replace with your MySQL password
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")  # Replace with your MySQL host
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")  # Replace with your MySQL port
POLYMARKET_DB = f"polymarket_{EVENT_ID}"
KALSHI_DB = f"kalshi_{EVENT_ID}"

# SQLAlchemy engine setup
POLYMARKET_ENGINE = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{POLYMARKET_DB}"
)
KALSHI_ENGINE = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{KALSHI_DB}"
)

# Helper function to get all table names (players/markets) from a database
def get_player_list(engine):
    try:
        with engine.connect() as conn:
            # Query information schema to get table names
            result = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = :db"),
                {"db": engine.url.database}
            )
            tables = [row[0] for row in result]
        
        # Extract player names, removing prefixes like 't4', 'num_5', etc.
        players = set()
        for table in tables:
            name = table
            # Remove common prefixes to get clean player names
            for prefix in ['t', 'num_']:
                if re.match(f"^{prefix}\d+", name):
                    name = name[len(prefix) + 1:]  # Remove prefix and number
            players.add(name)
        return sorted(list(players))
    except SQLAlchemyError as e:
        print(f"Error fetching player list: {e}")
        return []
'''
# Helper function to get market data for a specific player
def get_market_data(engine, player, yes_no, limit=100):
    try:
        with engine.connect() as conn:
            # Get all tables related to the player
            result = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = :db AND table_name LIKE :pattern"),
                {"db": engine.url.database, "pattern": f'%{player}%'}
            )
            tables = [row[0] for row in result]
        
        if not tables:
            return {"error": "No markets found for player"}
        
        # Union query to combine data from all relevant tables
        union_queries = []
        for table in tables:
            query = f"SELECT market_name, yes_price, no_price, trading_volume, timestamp FROM `{table}`"
            union_queries.append(query)
        
        query = " UNION ALL ".join(union_queries) + " ORDER BY timestamp DESC LIMIT :limit"
        with engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit})
            rows = result.fetchall()
        
        # Process data
        data = []
        for row in rows:
            market_name, yes_price, no_price, volume, timestamp = row
            price = yes_price if yes_no.lower() == 'yes' else no_price
            data.append({
                "market_name": market_name,
                "price": float(price) if price is not None else 0.0,
                "volume": str(volume) if volume is not None else "N/A",
                "timestamp": str(timestamp)
            })
        
        # Calculate metrics
        latest_row = data[0] if data else None
        metrics = {
            "volume": latest_row["volume"] if latest_row else "N/A",
            "bid_ask_spread": abs(latest_row["price"] - (1 - latest_row["price"])) if latest_row else "N/A",
            "order_volume": "N/A"  # Placeholder, as order volume isn't directly available
        }
        
        return {"data": data, "metrics": metrics}
    except SQLAlchemyError as e:
        print(f"Error fetching market data: {e}")
        return {"error": "Database error"}

@app.route('/')
def serve_index():
    return send_file('index.html')

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)