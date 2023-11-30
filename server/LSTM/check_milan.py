import pandas as pd

input_file = "datasets/out/milan/data"

# Read the CSV file line by line and identify lines with discrepancies
problematic_lines = []
with open(input_file, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        data = line.strip().split("\t")
        if len(data) != 6:
            problematic_lines.append((line_number, line))

# Print problematic lines and their line numbers
for line_number, line in problematic_lines:
    print(f"Line {line_number} has {len(data)} fields instead of 6: {line}")

# Read the CSV file, skipping lines with too many fields
df = pd.read_csv(input_file, sep="\t", header=None, names=["date", "time", "sensor", "value", "activity", "log"], error_bad_lines=False)

# Further data processing or analysis on the cleaned dataframe
# ...

print(df.head())  # Display the first few rows of the cleaned dataframe
