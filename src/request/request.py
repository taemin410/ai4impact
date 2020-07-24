import requests
import csv

PASSWORD = 1051579459
SUBMISSION_URL = "http://3.1.52.222/submit/pred"
RESULT_URL = "https://ai4impact.org/p003-log.csv"

def submit_answer(pred_val):
    payload = {
                'pwd': PASSWORD,
                'value': pred_val
            }

    res = requests.post(SUBMISSION_URL, data = payload)

    print("response", res.text)

def get_result_from_web():
    with requests.Session() as s:
        try:
            download = s.get(RESULT_URL)

            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')

            print("Successfully downloaded result from web!")
            return cr

        except Exception as err:
            print("Error in getting result from web: ", err)
            return None
        