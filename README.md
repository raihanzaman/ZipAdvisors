# ZipAdvisors

# ZipAdvisors

**ZipAdvisors** is a real-time arbitrage detection and analytics platform for prediction markets, focused on uncovering price discrepancies between **Kalshi** and **Polymarket**. Designed for hobbyist sports bettors, it combines high-frequency data scraping, machine learning, and advanced chart visualizations to provide actionable trading insights.

---

## 🔍 Project Overview

- **Arbitrage Monitoring**: Identifies statistically significant mispricings between yes/no contracts on Kalshi and Polymarket.
- **Custom Web Scrapers**: Built using `Selenium` and `requests`, the scrapers operate on ~20-second intervals to avoid API rate limits and capture up-to-date pricing data.
- **Data Pipeline**: Scraped data is sanitized, standardized, and stored in SQLite databases. Each market has its own table. SQLAlchemy is used to safely connect and retrieve data for the frontend.
- **Machine Learning Integration**: We use `XGBoost` models to forecast price movements using features like momentum, volatility, and spread. Results are displayed in real time on the web interface.
- **Real-Time Visualizations**: Built with Flask, HTML, CSS, and JavaScript, and powered by Plotly for interactive, chart-based overlays and forecasts.
- **Historical Analysis**: The platform includes a persistent database that enables backtesting, trend discovery, and strategy development.

---

## 🧰 Tech Stack

- **Backend**: Python, Selenium, Requests, SQLite, SQLAlchemy, XGBoost
- **Frontend**: Flask, HTML/CSS, JavaScript, Plotly
- **Deployment**: Runs on NYU-hosted virtual machines with scheduled batch jobs

---

## 🎯 Goals

- Provide bettors with faster and more info-dense visualizations than Kalshi or Polymarket
- Enable custom overlays like momentum indicators and machine learning predictions
- Deliver actionable arbitrage alerts based on real-time and historical market data
- Maintain scalability for adding new markets, trading bots, and ML models

---

## 🚀 Future Plans

- Add automated betting simulations
- Expand to other prediction markets
- Support multiple ML models for ensemble forecasting
- Enhance UI with user authentication and dashboard features

---

## 📂 Repository Structure

```bash
zipadvisors/
├── backend/
│   ├── scraper_kalshi.py
│   ├── scraper_polymarket.py
│   ├── data_pipeline.py
│   └── ml_model.py
├── frontend/
│   ├── static/
│   ├── templates/
│   └── app.py
├── database/
│   └── market_data.db
├── README.md
└── requirements.txt
```



Notes:

If you want to run locally, clone repo and switch the route on app.py. 

XGBoost plot only works for the NBA Championship market, and for prediction markets that are both in Kalshi and Polymarket (since you are using the Kalshi data to predict movement for Polymarket price). Only predicts the yes price. Green background means up, red means down or can't predict. Use threshold to indicate risk (0.5 is most risky, 0 is least risky prediction). 
