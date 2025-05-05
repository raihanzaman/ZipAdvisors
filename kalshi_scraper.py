#Make sure to download relevant dependencies: selenium, webdriver_manager, sqlalchemy
#This script scrapes the Kalshi website for the latest prices of the markets and stores them in a database
import time
import sys
from sqlalchemy import create_engine, text
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from urllib3.exceptions import ReadTimeoutError
from standardize_names import standardizeColumnNames
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_NAME =os.getenv('DB_NAME')

# Function to scrape data

def scrape_kalshi(url, event_id, driver, isRefreshed):
    table_name = "K_" + event_id
    table_name = table_name.replace("-", "_")
    try:
        if driver.current_url != url:
            driver.get(url)
        else:
            print("Already on the desired url no need to refresh")
    except Exception as e:
        print("Error loading page", e)
        return
    wait = WebDriverWait(driver, 60)
   
    new_element = wait.until(
        EC.presence_of_element_located((By.XPATH, "//div[@style='flex: 1 1 0%;']"))
        )
    
    button_elements = driver.find_elements(By.XPATH, "//div[@style='flex: 1 1 0%;']")
    try:
        more_market_button = button_elements[-1]
        print(more_market_button.text)
        first_clickable = more_market_button.find_element(By.XPATH, ".//*[self::button or self::a or self::span or @role='button' or @onclick]")
        if isRefreshed:
            first_clickable.click()
        time.sleep(2)
    except NoSuchElementException as e:
        print("No more markets button")
    except ElementNotInteractableException as e:
        print("More markets button not interactable")
    except Exception as e: 
        print("Other error", e)

    # Find all market tiles for the event
    markets_containers = driver.find_elements(By.XPATH, "//div[starts-with(@class, 'binaryMarketTile-0-1-')]")

    print(f"Found {len(markets_containers)} markets") # Print the number of markets found
    
    for market in markets_containers:
        market_name = market.find_element(By.XPATH, ".//span[contains(@class, 'lining-nums') and contains(@class, 'tabular-nums')]/div")
        
        #java-script to get the inner text of the element - some information is not directly accessible
        market_name_text = standardizeColumnNames(driver.execute_script("return arguments[0].innerText;", market_name))
    
        buttons = market.find_elements(By.TAG_NAME, "button")

        yes_price = 0.0
        no_price = 0.0

        for button in buttons: 
            if "Yes" in button.text:
                try:
                    yes_price = float(button.text.split()[-1].replace('¢', '')) / 100
                except ValueError:
                    print("Error converting yes price to float, will put as 0", button.text.split()[-1].replace('¢', ''))
                    yes_price = 0
            elif "No" in button.text:
                try:
                    no_price = float(button.text.split()[-1].replace('¢', '')) / 100
                except ValueError:
                    print("Error converting yes price to float, will put as 0", button.text.split()[-1].replace('¢', ''))
                    no_price = 0

        print(f"Event: {market_name_text} | Yes: {yes_price} | No: {no_price}")
        print("timestamp is", datetime.now())
        # Close the driver and database connection
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
                id INT PRIMARY KEY AUTO_INCREMENT,
                market_name TEXT,
                yes_price REAL,
                no_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''))

            conn.execute(
            text(f"""
                INSERT INTO {table_name} (market_name, yes_price, no_price)
                VALUES (:market_name, :yes_price, :no_price)
            """),
            {
                "market_name": market_name_text,
                "yes_price": yes_price,
                "no_price": no_price
            }
            )
    print("Scraping completed")
    

def initatizeKalshiScrape(event_id, event_url, iterations, driver):
    counter = 0
    while counter < iterations:
        
        #Don't refresh too fast, page should automatically change data because it is javascript based 
        if counter % 3 == 0:
            try:
                driver.refresh()
                scrape_kalshi(event_url, event_id, driver, True)
                counter += 1
            except ReadTimeoutError:
                print("Read time out error - restarting scraper")
                driver.quit()
                options = webdriver.ChromeOptions()
                prefs = {
                    "profile.managed_default_content_settings.images": 2
                }
                
                options.add_experimental_option("prefs", prefs)
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--headless")  # Add this line to make the browser headless
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.set_page_load_timeout(65)  # Sets a 3-minute timeout for page load.
                scrape_kalshi(event_url, event_id, driver, True)
                counter += 1
            except TimeoutError as ex:
                print("Time out exception at t-60 seconds from last printed timestamp")
                driver.quit()
                options = webdriver.ChromeOptions()
                prefs = {
                    "profile.managed_default_content_settings.images": 2
                }
                
                options.add_experimental_option("prefs", prefs)
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--headless")  # Add this line to make the browser headless
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.set_page_load_timeout(65)  # Sets a 3-minute timeout for page load.
                scrape_kalshi(event_url, event_id, driver, True)
                counter += 1
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Restarting application...")
                time.sleep(3)  # Optional: Add a delay before restarting
                print("Time out exception at t-60 seconds from last printed timestamp")
                driver.quit()
                options = webdriver.ChromeOptions()
                prefs = {
                    "profile.managed_default_content_settings.images": 2
                }
                
                options.add_experimental_option("prefs", prefs)
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--headless")  # Add this line to make the browser headless
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver.set_page_load_timeout(65)  # Sets a 3-minute timeout for page load.
                scrape_kalshi(event_url, event_id, driver, True)
                counter += 1
        else:
            time.sleep(10 + int(random.uniform(5,10)))
            scrape_kalshi(event_url, event_id, driver, False)
            counter += 1
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
    prefs = {
    "profile.managed_default_content_settings.images": 2
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless")  # Add this line to make the browser headless
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(65)  # Sets a 3-minute timeout for page load.
    initatizeKalshiScrape(event_id, event_url, iterations, driver)

if __name__ == "__main__":
    main()
