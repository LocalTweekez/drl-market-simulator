import csv
import pandas as pd

def read_positions_from_csv(file_path):
    positions = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Convert appropriate fields to float
            row['priceOpen'] = float(row['priceOpen'])
            row['priceClose'] = float(row['priceClose'])
            row['positionSL'] = float(row['positionSL'])
            row['positionTP'] = float(row['positionTP'])
            row['profit'] = float(row['profit'])
            positions.append(row)
    return positions

def check_trade_logic(positions):
    errors = []
    for position in positions:
        price_open = position['priceOpen']
        price_close = position['priceClose']
        trade_result = position['tradeResult']
        position_type = position['positionType']
        profit = position['profit']

        if position_type == "BUY":
            if trade_result == "WIN" and not (price_close > price_open):
                errors.append(position)
            elif trade_result == "LOSS" and not (price_close <= price_open):
                errors.append(position)
        elif position_type == "SELL":
            if trade_result == "WIN" and not (price_close < price_open):
                errors.append(position)
            elif trade_result == "LOSS" and not (price_close >= price_open):
                errors.append(position)
        
        # Check for cases where open and close are the same but the trade is marked as a win
        if price_open == price_close and trade_result == "WIN":
            errors.append(position)

    return errors

# Path to the CSV file containing the positions
file_path = 'results/positions.csv'

# Read positions from the CSV file
positions = read_positions_from_csv(file_path)

# Check the trade logic and print errors
errors = check_trade_logic(positions)

# Printing results
if errors:
    print("Errors found in the following positions:")
    for error in errors:
        print(error)
else:
    print("All trades follow the correct logic.")
