import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pyfiglet

# Display "Preprocessing Data" with ASCII art-style font
ascii_art = pyfiglet.figlet_format("Preprocessing Data", font="digital")
print(ascii_art)

# Read the original CSV file
data = pd.read_csv('top_10_per_label.csv')

# Convert the "datetime" column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract time-based features
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['second'] = data['datetime'].dt.second

# Extract the time portion and calculate time in seconds
data['time'] = data['datetime'].dt.time
data['time_seconds'] = data['time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)  # Convert time to seconds

# Apply StandardScaler on the time column
scaler = StandardScaler()
data['time_scaled'] = scaler.fit_transform(data[['time_seconds']])

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Wrap the data preprocessing and saving process with tqdm
for dataset, dataset_name in tqdm([(train_data, 'training_set'), (test_data, 'testing_set')]):
    # Save the preprocessed dataset to a CSV file (including one-hot encoded features)
    dataset_preprocessed = pd.get_dummies(dataset, columns=dataset.select_dtypes(include='object').columns)
    dataset_preprocessed.to_csv(f'preprocessed_{dataset_name}.csv', index=False)

    # Save the original unprocessed dataset to a CSV file
    dataset.to_csv(f'unprocessed_{dataset_name}.csv', index=False)
