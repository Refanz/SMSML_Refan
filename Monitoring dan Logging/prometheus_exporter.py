from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')

# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')

# Metrik Model
PREDICTION_FAILED_TOTAL = Counter('prediction_failed_total', 'Total number of failed predictions')


# Endpoint untuk Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik sistem setiap kali /metrics diakses
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# Endpoint untuk mengakses API model dan mencatat metrik
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()

    # Kirim request ke API model
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()

    try:
        response = requests.post(api_url, json=data)
        response.raise_for_status()

        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)

        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        PREDICTION_FAILED_TOTAL.inc()
        return jsonify({
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
