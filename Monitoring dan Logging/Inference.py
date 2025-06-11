import time

import requests
import json

MODEL_API_URL = "http://127.0.0.1:8003/predict"


def test_predict(columns: list, data: list):
    headers = {'Content-Type': 'application/json'}

    payload = {
        "dataframe_split": {
            "columns": columns,
            "data": data
        }
    }

    input_data = json.dumps(payload)

    try:
        response = requests.post(MODEL_API_URL, headers=headers, data=input_data)
        response.raise_for_status()

        time.sleep(3)

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"[RequestError]: {e}")
        return None
    except Exception as e:
        print(f"[Unexpected Error]: {e}")
        return None


if __name__ == '__main__':
    sample_columns = [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat"
    ]
    sample_data = [
        [
            0.6000000000000001,
            0.6666666666666666,
            0.2222222222222222,
            0,
            0.875,
            1,
            0,
            1,
            0,
            1,
            0,
            0.6666666666666666,
            0.3333333333333333,
            0.75,
            0.875,
            0,
            0.6666666666666666,
            0.5,
            0,
            0.875,
            0.8,
            0
        ],
        [
            1,
            0.6666666666666666,
            0.4444444444444444,
            0,
            0.25,
            1,
            0,
            1,
            0,
            1,
            0,
            0.3333333333333333,
            0.6666666666666666,
            0.875,
            0.875,
            0,
            0.6666666666666666,
            0.5,
            0,
            0.875,
            0.8,
            0.6666666666666666
        ],
        [
            0.4,
            1,
            0.2222222222222222,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            0.6666666666666666,
            0.6666666666666666,
            0.75,
            0.875,
            0,
            0.6666666666666666,
            0.5,
            0,
            0.875,
            0.8,
            0.3333333333333333
        ],
        [
            0.4,
            0,
            0.4444444444444444,
            1,
            0.625,
            1,
            0,
            0,
            0.8181818181818182,
            1,
            0.25,
            0.6666666666666666,
            0.6666666666666666,
            0.375,
            0.75,
            0,
            0.6666666666666666,
            0.5,
            1,
            0.375,
            0.8,
            0
        ],
        [
            0,
            1,
            1,
            1,
            0.375,
            1,
            0,
            0,
            0.3636363636363636,
            0,
            0.5,
            0.6666666666666666,
            0.6666666666666666,
            0.875,
            0.875,
            0,
            0.6666666666666666,
            0.5,
            1,
            0.375,
            0.4,
            0.5
        ]
    ]

    print(f"Test mengirimkan request ke {MODEL_API_URL}")
    prediction_result = test_predict(sample_columns, sample_data)

    if prediction_result is not None:
        print("[Prediction Result]")
        print(json.dumps(prediction_result, indent=4))
    else:
        print("Inferensi gagal!")