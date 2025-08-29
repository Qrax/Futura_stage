import csv
import numpy 
import os
# Open the file "Logboek_Quincy_2025_Juli.csv" Which is in the same folder as this python code
# Take the position from this script
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "Logboek_Quincy_2025_augustus.csv")

with open(file_path, "r") as file:
    reader = csv.reader(file)
    data = [row for row in reader]

# Convert the data to a NumPy array
data_array = numpy.array(data, dtype=object)

# Print the size (columns, rows)
print(data_array.shape)

'''
Look trough the file. Only save the following 2 rows :
If there is one of the following words in the row;
Maandag, Dinsdag, Woensdag, Donderdag, Vrijdag

Take the data from column 1 and 7 of that row.
'''

def extract_weekday_columns(data_array):
    weekdays = {"Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag"}
    result = []
    for row in data_array:
        # Convert row to list of strings for searching
        row_strs = [str(cell) for cell in row]
        if any(day in cell for cell in row_strs for day in weekdays):
            result.append((row[1], row[7]))
    return result

filtered_data = extract_weekday_columns(data_array)
print(filtered_data[10])

# Save this to a new .txt file that's saved in the same folder as this script. But put ======= between each row
output_file_path = os.path.join(script_dir, "filtered_data_augustus.txt")
with open(output_file_path, "w") as f:
    for item in filtered_data:
        f.write(",".join(map(str, item)) + "\n")
        f.write("===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================\n")
