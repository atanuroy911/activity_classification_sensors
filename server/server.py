from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from scipy.stats import mode
from datetime import datetime
import json


app = Flask(__name__)

# Later you can load the entire pipeline
loaded_model = joblib.load('random_forest_model.mod')

@app.route('/data', methods=['GET'])
def get_data():
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
        # print(selected_data)

        response_data = {'data': selected_data}
        
        final_data = json.dumps(response_data)
        # print(final_data)
        return final_data
        
        # print(f"Selected data: {selected_data}")
        # print(f"Prediction from server: {prediction}")
        
        break  # Break the loop after successful processing

# Load the MinMaxScaler

@app.route('/last-data', methods=['GET'])
def get_last_data():
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
            max(0, random_row_index) : random_row_index
        ]
        
        if len(selected_rows) < 1:
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
        # print(selected_data)

        response_data = {'data': selected_data}
        
        final_data = json.dumps(response_data)
        # print(final_data)
        return final_data
        
        # print(f"Selected data: {selected_data}")
        # print(f"Prediction from server: {prediction}")
        
        break  # Break the loop after successful processing

@app.route('/room-data', methods=['GET'])
def seg_data():
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
        # print(selected_data)

        selected_columns_bedroom = [0] + list(range(4, 11))  # Columns 0, 4 to 10
        selected_columns_diningroom = list(range(12, 18))  # Columns 0, 4 to 10

        formatted_data = {'bedroom': []}

        for entry in selected_data:
            bedroom_data = [entry[col] for col in selected_columns_bedroom]
            
            formatted_data['bedroom'].append([bedroom_data])
        # for entry in selected_data:
        #     diningroom_data = [entry[col] for col in selected_columns_diningroom]
            
        #     formatted_data['diningroom'].append([diningroom_data])

        print(formatted_data)

        # # print(data)       
        # response_data = {'data': selected_data}
        
        # final_data = json.dumps(response_data)
        # print(final_data)
        return formatted_data
        break

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print('Data Received')
        data = request.get_data()  # Get the JSON data from the request
        data = json.loads(data)
        data = data['data']
        print(data)
        # if 'data' in data:
        #     data_list = data['data']  # Access the 'data' key
        #     data_fin = json.loads(data_list)
        #     print(data)
        # df = pd.read_csv('new_dataset.csv')
        

        # Define the column names for the DataFrame
        column_names = ['time', 'D001', 'D002', 'D004', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031']

        
        # Convert the received data to a DataFrame
        data = pd.DataFrame(data, columns=column_names)

        # print(data)

        # Convert time column to datetime format with appropriate format strings
        data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S.%f', errors='coerce')
        data['time'] = data['time'].combine_first(pd.to_datetime(data['time'], format='%H:%M:%S', errors='coerce'))

        # Calculate the average time difference between consecutive timestamps
        time_diff = (data['time'].diff() / np.timedelta64(1, 's')).mean()

        # Fill missing 'time' values by adding the average time difference
        data['time'] = data['time'].fillna(method='ffill') + pd.to_timedelta(time_diff, unit='s')

        # Convert 'time' column to floating point seconds since start
        data['time'] = (data['time'] - data['time'].min()).dt.total_seconds()

        sensor_columns = data.columns[1:]  # Exclude 'time' and 'label'
        for column in sensor_columns:
            data[column] = data[column].apply(lambda x: 0 if x == 'OFF' else 1)
        data.to_csv('received.csv', index=False)

        y_pred = loaded_model.predict(data)

        # Find the majority class in the predictions
        # print(y_pred)
        majority_class = mode(y_pred)[0][0]
        label_encoder = joblib.load('label_encoder.mod')
        class_names = label_encoder.classes_

        # Get the majority class label
        prediction = class_names[majority_class]

        print("Majority Predicted Class Label:", prediction)

        # data.head
        # print(data)
        # Process the DataFrame as needed
        # For example, you can perform predictions or other operations here
        
        return jsonify({'message': 'Data received and processed successfully', 'prediction': prediction})
        # return "Success", 200  # Return a response
    except Exception as e:
        print(str(e))
        return "Error", 500  # Return an error response
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)
