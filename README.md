# ZipAdvisors

# Project Introduction: 
ZipAdvisors is developing an arbitrage monitoring software for prediction markets, focusing on 
identifying price discrepancies between Kalshi and Polymarket. The project consists of a backend data 
pipeline that collects data via a custom-built web scraper. We coded it with the selenium and requests 
library. The scrapper processes, and stores market data, coupled with analytical tools to detect trading 
opportunities. We chose a scrapper instead of an API to collect our data because an API call wouldn’t be 
as frequent as a real-time web scraper and gets rate limited fast. 
By systematically tracking yes/no contract prices across these platforms, the software aims to 
uncover mispricings that could yield risk-free profits through carefully balanced positions. This tool is 
designed for hobbyists who sports bet, providing them with real-time insights and historical trends to 
inform their betting strategies. 
# Overall Goal: 
The primary objective is to give the hobbyist bettor access to dynamic visualizations and machine 
learning algorithms to place better bets. We do this by providing ~20 second scraped price data and 
chart-based visualizations of price data that is faster and more info dense than the default charts provided 
by Polymarket and Kalshi, while also supporting custom overlays and momentum/price prediction 
forecasts for event contracts.  
We do so by programming our product to collect pricing data, identify statistically significant 
divergences between equivalent contracts, and generate actionable advice for traders. Beyond immediate 
arbitrage detection, the software also maintains a historical database to support volatility analysis, 
backtesting, and strategy development. Ultimately, the goal is to offer users a competitive edge by 
highlighting inefficiencies in these rapidly evolving markets. 
# Data Pipeline: 
The current implementation consists of dual-engine Python-based web scrapers—one for Kalshi, 
and another for Polymarket—both built using Selenium to handle dynamic content. These scrapers run on 
scheduled intervals via batch scripts, capturing market names, yes/no prices, and trading volumes. The 
data is standardized to ensure consistency and stored in remote MySQL databases, with each market maintaining 
its own table. A helper function sanitizes market names to prevent SQL injection and ensure compatibility 
with database naming conventions. We run the dual-engine scrapers on a VM (Virtual Machine) on 
NYU’s servers. 
Now, we have built the user Interface for the software, served on a web-server. The frontend 
architecture is composed of HTML, CSS, and JavaScript, wrapped in the Python web framework, Flask. 
We have used the professors database connection to store and retrieve data via SQLalchemy to our 
frontend. We are working on using the best charting library for our custom overlay component, and we 
estimate that will be the majority of work from this milestone submission to the final deadline
