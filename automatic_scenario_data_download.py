from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
from itertools import product

# Parameter lists (edit as needed)
counties = ['gavleborgs_lan']      # Add more counties as needed
indicators = ['medeltemperatur']   # Climate indicator, e.g., temperature
scenarios = ['rcp45']              # Emission scenarios
periods = ['2041-2070']            # Time periods
seasons = ['year']                 # e.g., 'year', 'summer', 'winter'
value_types = ['abs']              # 'abs' (absolute), 'anom' (anomaly)

# URL template for each parameter combination
url_template = (
    "https://www.smhi.se/en/climate/tools-and-inspiration/climate-change-scenario/"
    "climate-change-scenario-tool/met/{county}/{indicator}/{scenario}/{period}/{season}/{value_type}"
)

# Directory to save downloaded files
download_dir = r"F:\climate_data"
os.makedirs(download_dir, exist_ok=True)


chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True
})
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

# Start Chrome browser
driver = webdriver.Chrome(options=chrome_options)

for combo in product(counties, indicators, scenarios, periods, seasons, value_types):
    county, indicator, scenario, period, season, value_type = combo
    url = url_template.format(
        county=county,
        indicator=indicator,
        scenario=scenario,
        period=period,
        season=season,
        value_type=value_type
    )
    print(f" Processing: {url}")
    driver.get(url)

    # Handle cookie consent popup
    try:
        WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[class*='cky-btn-accept']"))
        ).click()
        print("  Cookie accepted")
        time.sleep(1)
    except Exception:
        pass

    # Wait for and click the download CSV button
    try:
        download_btn = WebDriverWait(driver, 16).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[title='Download graph data (.csv)']"))
        )
        download_btn.click()
        print("   Download button clicked")
    except Exception as e:
        print(f"   Download button not found, possibly no data for this combination: {e}")
        continue

    # Wait for the file to finish downloading
    time.sleep(8)

    # Rename the most recent CSV file to a meaningful name
    csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
    if csv_files:
        latest_file = max([os.path.join(download_dir, f) for f in csv_files], key=os.path.getctime)
        new_name = f"{county}_{indicator}_{scenario}_{period}_{season}_{value_type}.csv"
        new_path = os.path.join(download_dir, new_name)
        if not os.path.exists(new_path):
            os.rename(latest_file, new_path)
            print(f"  Renamed to: {new_name}")
        else:
            print(f"   File already exists: {new_name}")
    else:
        print("  No downloaded CSV file detected")

driver.quit()
print("\nAll downloads and renaming complete!")
