import requests
import pandas as pd
import random
from datetime import datetime

url = 'http://127.0.0.1:12345/predict'  # Server URL

# Load data from CSV file
data = pd.read_csv('new_dataset.csv')  # Replace with your CSV file
data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S.%f', errors='coerce')
data['time'] = data['time'].combine_first(pd.to_datetime(data['time'], format='%H:%M:%S', errors='coerce'))

# Get current hour and minute
current_time = datetime.now()
current_hour = current_time.hour
current_minute = current_time.minute

while True:
    # Select rows with matching hour and minute
    matching_rows = data[
        (data['time'].dt.hour == current_hour) &
        (data['time'].dt.minute == current_minute)
    ]
    
    if len(matching_rows) == 0:
        continue
    
    random_row = matching_rows.sample(n=1)
    
    # Get the index of the random row
    random_row_index = random_row.index[0]
    
    # Select the previous 20 rows (if available)
    selected_rows = data.loc[
        max(0, random_row_index - 20) : random_row_index
    ]
    
    if len(selected_rows) < 20:
        continue
    
    # Convert selected rows to a list of values
    # Convert the 'datetime' column to a pandas datetime object
    selected_rows['time'] = pd.to_datetime(selected_rows['time'])

    # Extract the time part and replace the 'datetime' column
    selected_rows['time'] = selected_rows['time'].dt.time
    
    selected_rows = selected_rows.iloc[::-1]

    # Convert the 'time' column to a string
    selected_rows['time'] = selected_rows['time'].astype(str)
    # selected_rows.drop(columns=['label_index'], inplace=True)
    selected_rows.to_csv('sent.csv', index=False)
    selected_data = selected_rows.drop(columns=['label']).values.tolist()[::-1]
    # print(selected_rows)
    
    # Send a POST request to the server
    response = requests.post(url, json={'data': selected_data})
    
    # Get the prediction from the server's response
    prediction = response.json()
    print(prediction)
    
    # print(f"Selected data: {selected_data}")
    # print(f"Prediction from server: {prediction}")
    
    break  # Break the loop after successful processing
