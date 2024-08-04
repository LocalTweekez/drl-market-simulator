import csv
import ast

def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        for line in infile:
            # Strip any leading/trailing whitespace from the line
            line = line.strip()
            # Extract the data between the brackets
            data_str = line.split('INPUT: [')[1].split('] | OUTPUT: []')[0]
            # Convert the data string to a list of floats
            data_list = ast.literal_eval(f'[{data_str}]')
            # Write the list as a row in the CSV
            writer.writerow(data_list)

# Call the function with the input and output filenames
process_file('new2.txt', 'output.csv')
