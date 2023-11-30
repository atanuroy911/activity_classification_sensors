import requests

url = 'http://127.0.0.1:12345/data'
def send_requests():
    r = requests.post(url)
    print(r.content)

if __name__ == '__main__':
    send_requests()