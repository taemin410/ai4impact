import requests

PASSWORD = 1051579459
URL = "http://3.1.52.222/submit/pred"

def submit_answer(pred_val):
    """
    @TODO: send this hourly OR depending on the input to the model -> should be confirmed after reaing the description
    """
    payload = {
                'pwd': PASSWORD,
                'value': pred_val
            }

    res = requests.post(URL, data = payload)

    print("response", res.text)



submit_answer(100)