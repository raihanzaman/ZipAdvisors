import time
from sqlalchemy import create_engine, text
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from standardize_names import standardizeColumnNames
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.common.exceptions import StaleElementReferenceException
from urllib3.exceptions import ReadTimeoutError
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import sys
import threading

from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME =os.getenv('DB_NAME')

# Function to scrape data

def run_with_timeout(func, args=(), kwargs={}, timeout=60):
    """
    Runs a function in a separate thread. If it hangs beyond `timeout` seconds, returns False.
    """
    result = [None]
    
    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            print("❌ Scrape error inside thread:", e)
            result[0] = False

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⚠️ Scrape thread timed out — likely a silent hang.")
        return False
    return result[0]


def scrape_polymarket(url, event_id, driver, last_scrape_info):
    try:
        driver.get(url)
    except Exception as e:
        print("Error loading page", e)
        return False
    wait = WebDriverWait(driver, 30)

    # Wait until the element is present in the DOM
    element = wait.until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'c-dhzjXW') and contains(@class, 'c-dhzjXW-ifipQUC-css')]"))
    )

    # Extract event name and standardize it
    table_name = "P_" + standardizeColumnNames(event_id)
    
    # Locate the container div for all markets
    markets_containers = driver.find_elements(By.XPATH, "//div[contains(@class, 'c-dhzjXW') and contains(@class, 'c-dhzjXW-ifipQUC-css')]")
    print(f"Found {len(markets_containers)} markets")
    
    for market in markets_containers:
        market_name = standardizeColumnNames(market.find_element(By.XPATH, ".//p[contains(@class, 'c-cZBbTr')]").text)
        try: 
            yes_price = market.find_element(By.XPATH, ".//div[contains(text(), 'Buy Yes')]").text.split()[-1].replace('¢', '')
        except StaleElementReferenceException:
            print("Stale element issue with yes, we will refresh and find it next time")
            return False
        try:
            no_price = market.find_element(By.XPATH, ".//div[contains(text(), 'Buy No')]").text.split()[-1].replace('¢', '')
        except StaleElementReferenceException:
            print("Stale element issue with no, we will refresh and find it next time")
            return False

        try:
            this_trading_volume = market.find_element(By.XPATH, ".//p[contains(., 'Vol.')]").text
        except StaleElementReferenceException:
            print("Stale element issue with trading volume, we will refresh and find it next time")
            return False

        this_trading_volume = this_trading_volume.split()[0].replace('$', '').replace(',', '')
        try:
            this_trading_volume = float(this_trading_volume)
        except ValueError:
            print("Error converting trading volume to float, will have strings", this_trading_volume)
            
        print(this_trading_volume)
        print(market_name)
        # Convert to float
        yes_price = float(yes_price) / 100
        no_price = float(no_price) / 100
        
        print(f"Event: {market_name} | Volume: {this_trading_volume} | Yes: {yes_price} | No: {no_price}")
        print(datetime.now())

        conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
            user=f'{DB_USER}',
            password=f'{DB_PASS}',
            host = 'jsedocc7.scrc.nyu.edu',
            port     = 3306,
            encoding = 'utf-8',
            db = f'{DB_NAME}'
        )
        engine = create_engine(conn_string)

        with engine.begin() as conn:
            conn.execute(text(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                market_name VARCHAR(255),
                trading_volume VARCHAR(255),
                yes_price FLOAT,
                no_price FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''))

            conn.execute(
            text(f"""
                INSERT INTO {table_name} (market_name, trading_volume, yes_price, no_price)
                VALUES (:market_name, :trading_volume, :yes_price, :no_price)
            """),
            {
                "market_name": market_name,
                "trading_volume": this_trading_volume,
                "yes_price": yes_price,
                "no_price": no_price
            }
            )
    last_scrape_info['last'] = datetime.now()
    print("Scraping completed")
    return True
    
def polymarket_scraper(event_url, event_id, iterations, driver, last_scrape_info):
    print("Starting scraper for", iterations, "iterations")
    counter = 0
    max_idle_time = timedelta(minutes=3)

    while counter < iterations:
        now = datetime.now()
        time_since_last = now - last_scrape_info['last']

        # Restart driver if idle too long
        if time_since_last > max_idle_time:
            print(f"⚠️ No successful scrape in {time_since_last}. Restarting driver...")
            driver.quit()
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--headless")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.set_page_load_timeout(65)
            last_scrape_info['last'] = datetime.now()

        print(f"\n⏳ Scraping iteration {counter+1} at {datetime.now()}")

        # Run with timeout protection
        success = run_with_timeout(
            scrape_polymarket,
            args=(event_url, event_id, driver, last_scrape_info),
            timeout=60  # seconds
        )

        if success:
            counter += 1
        else:
            print("⚠️ Scrape failed or timed out — restarting driver")
            driver.quit()
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--headless")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.set_page_load_timeout(65)

        time.sleep(10 + int(random.uniform(5, 10)))

    driver.quit()


def main():
    if len(sys.argv) < 3:
        print("Please provide an event url and a number of iterations (this scraper refreshes every ~30 seconds)" )
        return
    else:
        event_url = ""
        iterations = 0
        try:
            event_url = sys.argv[1]
            iterations = int(sys.argv[2])
            event_id = event_url.split("/")[-1]
        except ValueError:
            print("Please provide a valid event ID")
        except IndexError:
            print("Please provide a valid number of iterations")
        except Exception as e:  
            print("Error", e)
        if not event_url:
            print("Please provide a valid event ID")
            return
        if not iterations:
            print("Please provide a valid number of iterations")
        if iterations <= 0 or iterations > 9999:
            print("Please provide a valid number of iterations, or not too many")
            return
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Add this line to make the browser headless
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    last_scrape_info = {'last': datetime.now()}
    polymarket_scraper(event_url, event_id, iterations, driver, last_scrape_info)


if __name__ == "__main__":
    main()
