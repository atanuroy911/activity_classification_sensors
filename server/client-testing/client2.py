import requests
import pandas as pd
import random
from datetime import datetime

p_url = 'http://127.0.0.1:12345/predict'  # Server URL
d_url = 'http://127.0.0.1:12345/data'  # Server URL
r_url = 'http://127.0.0.1:12345/room-data'  # Server URL
l_url = 'http://127.0.0.1:12345/last-data'  # Server URL


def get_data(url):
    """Get data from the server"""
    r = requests.get(url)
    
    return r

def get_prediction(url, data):
    # Send a POST request to the server
    response = requests.post(url, json={'data': data})
    
    # Get the prediction from the server's response
    prediction = response.json()
    return prediction

def get_room(url):
    # Send a POST request to the server
    response = requests.get(url)
    
    # Get the prediction from the server's response
    room = response.json()
    return room
def get_last_data(url):
    # Send a POST request to the server
    response = requests.get(url)
    
    # Get the prediction from the server's response
    room = response.json()
    return room

if __name__ == '__main__':
    response = get_data(d_url)
    print("Get Data Response: ", response)
    # prediction = get_prediction(p_url, response.json())
    # print("Get Prediction Response: ", prediction)
    room = get_room(r_url)
    print("Get Room Data Response: ", room)

    response = get_last_data(l_url)
    print("Get Last Data Response: ", response)
