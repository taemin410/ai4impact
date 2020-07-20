import requests

PASSWORD = 1051579459
URL = "http://3.1.52.222/submit/pred"

def submit_answer(pred_val):
    payload = {
                'pwd': PASSWORD,
                'value': pred_val
            }

    res = requests.post(URL, data = payload)
    print("response", res.text)

def test_get_method():
    res = requests.get('http://localhost:4000/users')
    print(res.content)

