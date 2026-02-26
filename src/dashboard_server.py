from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
import psutil
import subprocess

app = Flask(__name__, static_folder='static')
CORS(app)

# Global data structure to be updated by the orchestrator
status_data = {
    'models': {},
    'system': {
        'cpu': 0,
        'gpu_util': '0%',
        'vram': '0/0 MB'
    }
}

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/status')
def get_status():
    # Update system stats
    status_data['system']['cpu'] = psutil.cpu_percent()
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        gpu_util, mem_used, mem_total = output.split(',')
        status_data['system']['gpu_util'] = f"{gpu_util.strip()}%"
        status_data['system']['vram'] = f"{mem_used.strip()}/{mem_total.strip()} MB"
    except:
        pass
    
    # Convert DictProxy to standard dict for JSON serialization
    response_data = {
        'models': dict(status_data['models']),
        'system': status_data['system']
    }
    
    return jsonify(response_data)

def start_dashboard(shared_dict):
    """
    Function to start the flask app. 
    It will link status_data['models'] to the shared_dict from multiprocessing.
    """
    global status_data
    status_data['models'] = shared_dict
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
