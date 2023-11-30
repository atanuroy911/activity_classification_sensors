import pandas as pd

# Load the preprocessed CSV file
data = pd.read_csv('aruba-bysecs-full.csv')

# Group the data by 'label_index' and select the top 10 rows from each group
top_10_per_label = data.groupby('label_index').head(10)

# Save the new dataset to a CSV file
top_10_per_label.to_csv('top_10_per_label.csv', index=False)
