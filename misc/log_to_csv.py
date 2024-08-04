import csv
import re
from datetime import datetime

def convert_to_csv(input_filename, output_filename, encoding='utf-8'):
    # Define the columns for the CSV
    columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    
    # Open the input and output files
    with open(input_filename, 'r', encoding=encoding) as infile, open(output_filename, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(columns)  # Write the header

        ctr = 0
        for i, line in enumerate(infile):
            print(i)
            try:
                # Exclude the first 39 characters from each line
                processed_line = (line[40:50]+","+ line[62:]).replace('"', '')
                print(processed_line)
                parts = processed_line.split(',')
                csv_writer.writerow(parts)
                ctr = i
            except Exception as e:
                print(f"Error processing line: {line}\nException: {e}")


def clean_field(field):
    # Remove newlines and extra spaces within the field
    return re.sub(r'[\n"]+', ' ', field).strip()

def reformat_date(date_str):
    try:
        # Convert date from YYYY.MM.DD to YYYY-MM-DD
        date_obj = datetime.strptime(date_str, '%Y.%m.%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        # Return the original string if it doesn't match the expected format
        return date_str

def clean_csv_file(input_file: str, output_file: str):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        
        # List to store cleaned rows
        cleaned_rows = []
        
        for row in reader:
            # Clean each field
            cleaned_row = [clean_field(cell) for cell in row]
            
            # Reformat the date (assuming the date is always in the first column)
            if cleaned_row:  # Ensure the row is not empty
                cleaned_row[0] = reformat_date(cleaned_row[0])
            
            cleaned_rows.append(cleaned_row)
    
    # Write cleaned rows to the output file
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)

# Example usage
input_file = "ohlcv.txt"
output_file = "output_1.csv"
output_file2 = "output_2.csv"

#convert_to_csv(input_file, output_file)
clean_csv_file(output_file, output_file2)