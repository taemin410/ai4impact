import requests
import threading

PASSWORD = 1051579459
URL = "http://3.1.52.222/submit/pred"
RESUBMISSION_TIME_INTERVAL = 3600

def submit_answer(pred_val):
    payload = {
                'pwd': PASSWORD,
                'value': pred_val
            }

    res = requests.post(URL, data = payload)

    print("response", res.text)

def run_submission_session():
    threading.Timer(RESUBMISSION_TIME_INTERVAL, run_submission_session).start()
    
    #@TODO: get pred val from somewhere else...
    pred_val = 10
    submit_answer(pred_val)

run_submission_session()