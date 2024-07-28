import requests
from bs4 import BeautifulSoup
import csv

# URL of the webpage to scrape
url = "https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the webpage content
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find the table
    table = soup.find("table")
    
    # Find all rows in the table
    rows = table.find_all("tr")
    
    # Open a CSV file to write the data
    with open("error_codes.csv", "w", newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Constant", "Code", "Description"])
        
        # Iterate through the rows and write the data to the CSV file
        for row in rows[1:]:  # Skip the header row
            cols = row.find_all("td")
            cols = [col.text.strip() for col in cols]
            writer.writerow(cols)
            
    print("Data has been written to error_codes.csv")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
