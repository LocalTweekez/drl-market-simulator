import csv

# Define the input CSV file and output text file
input_csv_file = "error_codes.csv"
output_text_file = "formatted_error_codes.txt"

# Open the input CSV file for reading
with open(input_csv_file, "r") as csv_file:
    reader = csv.reader(csv_file)
    # Skip the header row
    next(reader)
    
    # Open the output text file for writing
    with open(output_text_file, "w") as txt_file:
        # Iterate over each row in the CSV file
        for row in reader:
            constant, code, description = row
            # Format the line as specified
            formatted_line = f'if(code == {code}) {{\nreturn "{constant}: {description}"; }}\n'
            # Write the formatted line to the text file
            txt_file.write(formatted_line)

print(f"Data has been reformatted and written to {output_text_file}")
