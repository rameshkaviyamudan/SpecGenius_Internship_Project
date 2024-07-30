# -*- coding: utf-8 -*-
import asyncio
import platform
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO, emit
import os
import subprocess
import shutil
import json
import json
import os
import signal
import sys
import pandas as pd
import io
import numpy as np
import re
from flask import redirect
from flask import request
from flask import session
from flask import url_for
from flask import make_response
from flask import Flask, jsonify
from threading import Thread
from time import sleep
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)

#socketio = SocketIO(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
fine_tuned_model_id = os.environ.get('FINE_TUNED_MODEL_ID')
FINE_TUNED_MODEL_ID = fine_tuned_model_id if fine_tuned_model_id else "ft:gpt-3.5-turbo-0125:personal::9ZJtzJtN:ckpt-step-72"
MAIN_CSV_FILE = os.path.join('output', 'specifications', 'mega_combined.csv')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Directories and configurations
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
OUTPUT_FOLDER = os.path.join(UPLOAD_FOLDER, 'output')
UPLOADED_EVALUATION_FILE = None
LAST_EVALUATION_RESULT_FILE = 'last_evaluation_result.json'
MODEL_CONFIG_FILE = 'model_config.json'
CSV_HEADERS_FILE = 'csv_headers.json'
EVALUATION_RESULTS_FILE = 'evaluation_results.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SECRET_KEY'] = '1234'
app.config['SESSION_TYPE'] = 'filesystem'

import csv
def ensure_mega_combined_exists():
    if not os.path.exists(MAIN_CSV_FILE):
        print(f"mega_combined.csv not found. Creating an empty file at {MAIN_CSV_FILE}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(MAIN_CSV_FILE), exist_ok=True)
        
        # Create an empty file
        open(MAIN_CSV_FILE, 'w').close()
        print(f"Created empty mega_combined.csv file at {MAIN_CSV_FILE}")
    else:
        print(f"mega_combined.csv found at {MAIN_CSV_FILE}")

# Call this function early in your application startup
ensure_mega_combined_exists()
def ensure_json_files_exist():
    json_files = [
        LAST_EVALUATION_RESULT_FILE,
        MODEL_CONFIG_FILE,
        CSV_HEADERS_FILE,
        EVALUATION_RESULTS_FILE
    ]

    for file_name in json_files:
        file_path = os.path.join(BASE_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"{file_path} not found. Creating empty file.")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create an empty file
            with open(file_path, 'w') as f:
                f.write('')
            print(f"Created empty file at {file_path}.")
        else:
            print(f"{file_path} found.")

# Call this function early in your application startup
ensure_json_files_exist()
# Modify the existing get_csv_headers function to handle empty files
def get_csv_headers(csv_file_path):
    ensure_mega_combined_exists()
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            
            # Read the first row
            first_row = next(reader, None)
            
            if first_row is None:
                return []  # File is empty
            
            # Check if the first row contains '''
            if any("'''" in cell for cell in first_row):
                # If it does, use the second row as headers
                headers = next(reader, None)
            else:
                # If not, use the first row as headers
                headers = first_row
            
        return headers if headers else []
    except csv.Error:
        print(f"Error reading CSV file: {csv_file_path}")
        return []
        
    return headers if headers else []
def get_saved_headers():
    try:
        with open('csv_headers.json', 'r') as f:
            data = json.load(f)
            return data.get('headers', [])
    except FileNotFoundError:
        return []

# Assuming you have the CSV file path
csv_file_path = os.path.join('output', 'specifications', 'mega_combined.csv')
headers = get_csv_headers(csv_file_path)


def get_current_model_id():
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
            return config.get('model_id', "ft:gpt-3.5-turbo-0125:personal::9ZJtzJtN:ckpt-step-72")
    except FileNotFoundError:
        return "ft:gpt-3.5-turbo-0125:personal::9ZJtzJtN:ckpt-step-72"

def set_current_model_id(model_id):
    with open('model_config.json', 'w') as f:
        json.dump({'model_id': model_id}, f)

@app.route('/model-name', methods=['GET'])
def get_fine_tuned_model_name():
    return jsonify({"modelName": get_current_model_id()})

@app.route('/main-csv-headers', methods=['GET'])
def get_main_csv_headers():
    try:
        with open('csv_headers.json', 'r') as f:
            data = json.load(f)
            headers = data.get('headers', [])
        
        if not headers:
            return jsonify({"error": "No headers found. Please train your model first."}), 404
        
        return jsonify({"headers": headers})
    except FileNotFoundError:
        return jsonify({"error": "No headers found. Please train your model first."}), 404
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/get-csv-data', methods=['GET'])
def get_csv_data():
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        # Clean column names
        def clean_header(header):
            # Remove any non-ASCII characters
            header = re.sub(r'[^\x00-\x7F]+', '', header)
            # Remove special characters and extra spaces
            header = re.sub(r'[^a-zA-Z0-9\s]', '', header)
            # Replace spaces with underscores and convert to lowercase
            header = header.strip().replace(' ', '_').lower()
            return header

        df.columns = [clean_header(col) for col in df.columns]
        
        # Remove the first row if it's empty
        if df.iloc[0].isna().all() or (df.iloc[0] == '').all():
            df = df.iloc[1:].reset_index(drop=True)
        
        # If the dataframe is now empty, return an appropriate message
        if df.empty:
            return jsonify({'error': 'The CSV file is empty after cleaning'}), 400
        
        # Replace NaN values with None
        df = df.where(pd.notnull(df), None)
        
        # Convert DataFrame to a dictionary
        csv_data = {
            'headers': df.columns.tolist(),
            'rows': df.to_dict(orient='records')
        }
        
        # Custom JSON encoder to handle any remaining non-standard types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if pd.isna(obj):
                    return None
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, (np.datetime64, pd.Timestamp)):
                    return obj.isoformat()
                return super().default(obj)

        # Use the custom encoder to convert the data to JSON
        json_data = json.dumps(csv_data, cls=CustomEncoder)
        
        print("CSV data to be sent (first 1000 characters):", json_data[:1000])
        print("CSV data to be sent (last 1000 characters):", json_data[-1000:])
        
        return json_data, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error in get_csv_data: {str(e)}")
        return json.dumps({'error': f'An error occurred: {str(e)}'}), 500, {'Content-Type': 'application/json'}


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "OK"}), 200

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    # Handle PDF file upload
    pdf_file = request.files['pdf_file']
    # Save the PDF file to the desired location
    pdf_file_path = os.path.join(BASE_DIR, 'uploads', pdf_file.filename)
    pdf_file.save(pdf_file_path)

    # Start processing in a background thread
    Thread(target=process_file_background, args=(pdf_file_path, pdf_file.filename)).start()

    # Immediately return a response
    return jsonify({'message': 'PDF received and processing started'}), 202

def process_file_background(pdf_file_path, filename):
    try:
        process_file(pdf_file_path, filename, run_notebook=True)
        socketio.emit('processing_complete', {'message': 'PDF processing completed successfully'})
    except Exception as e:
        socketio.emit('processing_error', {'message': f'Error processing PDF: {str(e)}'})



# Create directories if they do not exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message_to_browser', {'data': 'Client connected successfully'}, broadcast=True)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('csv_data')
def handle_csv_data(csv_data):
    emit('csv_data', {'data': csv_data}, broadcast=True)

    
def emit_progress(message):
    print(f"Emitting progress message: {message}")  # Debug message
    socketio.emit('progress', {'message': message})
    socketio.emit('progress_message', {'message': message})



def process_file(filepath, filename, run_notebook=False):
    command = f'nougat pdf "{filepath}" --out "{OUTPUT_FOLDER}" --recompute --no-skipping'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running nougat command: {result.stderr}")
    else:
        print(f"Nougat command output: {result.stdout}")
        emit_progress("Extraction from PDF done")
        
        if run_notebook:
            run_notebook_process(filename)

def run_notebook_process(filename):
    notebook_path = os.path.join(BASE_DIR, 'using_nougatipynb.ipynb')

    # Set environment variables for the notebook
    os.environ['NOUGAT_PDF_FILENAME'] = filename
    os.environ['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    os.environ['PROCESSED_FOLDER'] = PROCESSED_FOLDER
    os.environ['OUTPUT_FOLDER'] = OUTPUT_FOLDER

    command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output {notebook_path} --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.allow_errors=True"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"Notebook execution output: {output.strip()}")

    stderr = process.stderr.read()
    if stderr:
        print(f"Notebook execution error: {stderr}")
    else:
        # Get the CSV data from the notebook output
        csv_data = process.stdout.read()

        # Emit the CSV data to the browser
        emit('csv_data', {'data': csv_data}, broadcast=True)

@app.route('/')
def index():
    return render_template('index.html', headers=headers, encoding='utf-8')

@app.route('/restart', methods=['POST'])
def restart():
    print("Restart requested. Shutting down...")
    # Shutdown the server
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server is restarting...', 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        process_file(filepath, filename)
        return redirect(url_for('process_notebook', filename=filename))
    return redirect(request.url)

def run_notebook(partition_data):
    command = [
        'jupyter', 'nbconvert', 
        '--to', 'notebook', 
        '--execute', 'finetune.ipynb', 
        '--output', 'finetune_output.ipynb',
        '--ExecutePreprocessor.timeout=-1',  # No timeout
        f'--param', f'partition_data={partition_data}'
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    for line in iter(process.stdout.readline, b''):
        print(line.decode().strip())
        socketio.emit('progress', {'message': line.decode().strip()})
    
    for line in iter(process.stderr.readline, b''):
        print(line.decode().strip())
        socketio.emit('progress', {'message': f"Error: {line.decode().strip()}"})
    
    process.wait()
    
    if process.returncode != 0:
        socketio.emit('progress', {'message': 'Fine-tuning failed'})
    else:
        socketio.emit('progress', {'message': 'Fine-tuning completed successfully'})

# Assuming you have these configurations
ALLOWED_EXTENSIONS = {'csv'}
EVALUATION_FOLDER = 'evaluation_data'

if not os.path.exists(EVALUATION_FOLDER):
    os.makedirs(EVALUATION_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Global variable to store the uploaded evaluation file path
UPLOADED_EVALUATION_FILE = None

@app.route('/upload-evaluation', methods=['POST'])
def upload_evaluation():
    global UPLOADED_EVALUATION_FILE

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(EVALUATION_FOLDER, filename)
        UPLOADED_EVALUATION_FILE = filepath  # Store the file path globally

        file.save(filepath)
        print(f"File uploaded: {filepath}")

        # Process the uploaded CSV file
        try:
            # Load the CSV file
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Clean column names
            def clean_header(header):
                # Remove any non-ASCII characters
                header = re.sub(r'[^\x00-\x7F]+', '', header)
                # Remove special characters and extra spaces
                header = re.sub(r'[^a-zA-Z0-9\s]', '', header)
                # Replace spaces with underscores and convert to lowercase
                header = header.strip().replace(' ', '_').lower()
                return header

            df.columns = [clean_header(col) for col in df.columns]
            
            # Remove the first row if it's empty
            if df.iloc[0].isna().all() or (df.iloc[0] == '').all():
                df = df.iloc[1:].reset_index(drop=True)
            
            # If the dataframe is now empty, return an appropriate message
            if df.empty:
                return jsonify({'error': 'The CSV file is empty after cleaning'}), 400
            
            # Replace NaN values with None
            df = df.where(pd.notnull(df), None)
            
            # Convert DataFrame to a dictionary
            csv_data = {
                'headers': df.columns.tolist(),
                'rows': df.to_dict(orient='records')
            }
            
            # Custom JSON encoder to handle any remaining non-standard types
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if pd.isna(obj):
                        return None
                    if isinstance(obj, (np.integer, np.floating, np.bool_)):
                        return obj.item()
                    if isinstance(obj, (np.datetime64, pd.Timestamp)):
                        return obj.isoformat()
                    return super().default(obj)

            # Use the custom encoder to convert the data to JSON
            json_data = json.dumps(csv_data, cls=CustomEncoder)
            
            print("CSV data to be sent (first 1000 characters):", json_data[:1000])
            print("CSV data to be sent (last 1000 characters):", json_data[-1000:])
            
            return json_data, 200, {'Content-Type': 'application/json'}
        except Exception as e:
            print(f"Error in upload-evaluation: {str(e)}")
            return json.dumps({'error': f'An error occurred: {str(e)}'}), 500, {'Content-Type': 'application/json'}
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/get-evaluation-data', methods=['GET'])
def get_evaluation_data():
    try:
        evaluation_file = os.path.join('evaluation_data.csv')
        if not os.path.exists(evaluation_file):
            return jsonify({'error': 'Evaluation data not found'}), 404
        
        # Load the CSV file
        df = pd.read_csv(evaluation_file, encoding='utf-8')
        
        # Clean column names
        def clean_header(header):
            # Remove any non-ASCII characters
            header = re.sub(r'[^\x00-\x7F]+', '', header)
            # Remove special characters and extra spaces
            header = re.sub(r'[^a-zA-Z0-9\s]', '', header)
            # Replace spaces with underscores and convert to lowercase
            header = header.strip().replace(' ', '_').lower()
            return header

        df.columns = [clean_header(col) for col in df.columns]
        
        # Remove the first row if it's empty
        if df.iloc[0].isna().all() or (df.iloc[0] == '').all():
            df = df.iloc[1:].reset_index(drop=True)
        
        # If the dataframe is now empty, return an appropriate message
        if df.empty:
            return jsonify({'error': 'The CSV file is empty after cleaning'}), 400
        
        # Replace NaN values with None
        df = df.where(pd.notnull(df), None)
        
        # Convert DataFrame to a dictionary
        evaluation_data = {
            'headers': df.columns.tolist(),
            'rows': df.to_dict(orient='records')
        }
        
        # Custom JSON encoder to handle any remaining non-standard types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if pd.isna(obj):
                    return None
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, (np.datetime64, pd.Timestamp)):
                    return obj.isoformat()
                return super().default(obj)

        # Use the custom encoder to convert the data to JSON
        json_data = json.dumps(evaluation_data, cls=CustomEncoder)
        
        print("Evaluation data to be sent (first 1000 characters):", json_data[:1000])
        print("Evaluation data to be sent (last 1000 characters):", json_data[-1000:])
        
        return json_data, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"Error in get_evaluation_data: {str(e)}")
        return json.dumps({'error': f'An error occurred: {str(e)}'}), 500, {'Content-Type': 'application/json'}

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    try:
        # Get the evaluation data from the request
        data = request.get_json()
        evaluation_data = data.get('evaluationData')
        
        if evaluation_data == 'use_partitioned_data':
            print("Using partitioned data for evaluation")
            # Use the existing evaluation_data.csv file for partitioned data scenario
            eval_data_path = os.path.join('evaluation_data.csv')
            if not os.path.exists(eval_data_path):
                return jsonify({'error': 'Evaluation data not found. Please upload evaluation data first.'}), 400
        else:
            print("Using uploaded data for evaluation")
            # Retrieve the uploaded evaluation file path from session
            if not UPLOADED_EVALUATION_FILE or not os.path.exists(UPLOADED_EVALUATION_FILE):
                return jsonify({'error': 'Uploaded evaluation data not found. Please upload evaluation data first.'}), 400
            
            eval_data_path = UPLOADED_EVALUATION_FILE
        
        os.environ['EVAL_DATA_PATH'] = eval_data_path

        # Path to the evaluation notebook
        eval_notebook_path = os.path.join(BASE_DIR, 'evaluate.ipynb')
        
        def execute_notebook():
            try:
                # Set environment variable for evaluation data path
                env = os.environ.copy()
                env['EVAL_DATA_PATH'] = eval_data_path

                command = f"jupyter nbconvert --to notebook --execute {eval_notebook_path} --output {eval_notebook_path} --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1"
                
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                
                for line in iter(process.stdout.readline, ''):
                    if line:
                        socketio.emit('eval_progress', {'message': line.strip()})
                
                rc = process.wait()
                if rc != 0:
                    error = process.stderr.read()
                    socketio.emit('eval_error', {'message': f"Error: {error}"})
                else:
                    # Assuming the evaluation results are saved in a file
                    with open('evaluation_results.json', 'r') as f:
                        results = json.load(f)


                    # Save the results as the last evaluation result
                    with open(LAST_EVALUATION_RESULT_FILE, 'w') as f:
                        json.dump(results, f)

                    socketio.emit('eval_complete', {'message': 'Evaluation completed successfully', 'results': results})
            except Exception as e:
                socketio.emit('eval_error', {'message': f"Error: {str(e)}"})

        # Start execution of the evaluation notebook in a separate thread
        Thread(target=execute_notebook).start()
        
        return jsonify({'message': 'Evaluation process started'}), 202

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# You might also want to add a route to get the evaluation results
@app.route('/get-evaluation-results', methods=['GET'])
def get_evaluation_results():
    try:
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
        return jsonify(results), 200
    except FileNotFoundError:
        return jsonify({'error': 'Evaluation results not found. Please run the evaluation first.'}), 404
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/download-evaluation-report', methods=['GET'])
def download_evaluation_report():
    try:
        with open('evaluation_results.json', 'r') as f:
            report_data = json.load(f)
        
        # If you want to format the JSON for better readability
        formatted_report = json.dumps(report_data, indent=2)
        
        return send_file(
            io.BytesIO(formatted_report.encode()),
            mimetype='application/json',
            as_attachment=True,
            download_name='evaluation_report.json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-last-evaluation-result', methods=['GET'])
def get_last_evaluation_result():
    try:
        file_path = os.path.join(BASE_DIR, LAST_EVALUATION_RESULT_FILE)
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Check if the file is empty
        if not content:
            return jsonify(None), 200
        
        # Try to parse the content as JSON
        last_result = json.loads(content)
        
        # Check if the parsed result is null or empty
        if last_result is None or (isinstance(last_result, dict) and not last_result):
            return jsonify(None), 200
        
        return jsonify(last_result), 200
    except FileNotFoundError:
        return jsonify(None), 200
    except json.JSONDecodeError:
        # Handle case where JSON is invalid
        return jsonify(None), 200
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/finetune', methods=['POST'])
def finetune_notebook():
    data = request.json
    partition_data = data.get('partitionData', False)

    finetune_notebook_path = os.path.join(BASE_DIR, 'finetune.ipynb')

    # Set the partition_data as an environment variable
    env = os.environ.copy()
    env['PARTITION_DATA'] = str(partition_data).lower()

    command = f"jupyter nbconvert --to notebook --execute {finetune_notebook_path} --output {finetune_notebook_path} --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.allow_errors=True --ExecutePreprocessor.timeout=-1"
    
    def execute_notebook():
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                socketio.emit('progress', {'message': output.strip()})
        
        rc = process.poll()
        
        if rc != 0:
            error = process.stderr.read()
            print(f"Error: {error}")
            socketio.emit('progress', {'message': f"Error: {error}"})
            return jsonify({'error': error}), 500
        else:
            socketio.emit('progress', {'message': 'Fine-tuning completed successfully'})
            return jsonify({'message': 'Fine-tuning completed successfully'}), 200

    # Start the notebook execution in a separate thread
    thread = Thread(target=execute_notebook)
    thread.start()

    return jsonify({'message': 'Fine-tuning process started'}), 202


    
@app.route('/query-model-for-evaluation', methods=['POST'])
def query_model_for_evaluation():
    eval_type = request.json.get('eval_type')
    eval_file = request.files.get('eval_file')

    if eval_type == 'partitioned':
        eval_file_path = os.getenv('PARTITION_FILE', None)
        if not eval_file_path or not os.path.exists(eval_file_path):
            return jsonify({'error': 'Partitioned data not found'}), 400
    elif eval_type == 'uploaded' and eval_file:
        eval_file_path = os.path.join(UPLOAD_FOLDER, eval_file.filename)
        eval_file.save(eval_file_path)
    else:
        return jsonify({'error': 'Invalid evaluation type or file missing'}), 400

    # Run the evaluation notebook
    command = f"jupyter nbconvert --to notebook --execute evaluation.ipynb --output evaluation_output.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.allow_errors=True"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output, error = process.communicate()

    if error:
        return error, 500
    else:
        # Clean up the partitioned file if it was used
        if eval_type == 'partitioned' and eval_file_path:
            os.remove(eval_file_path)

        return output, 200

@app.route('/process/<filename>')
def process_notebook(filename):
    notebook_path = os.path.join(BASE_DIR, 'using_nougatipynb.ipynb')
    
    # Set environment variable for the notebook
    os.environ['NOUGAT_PDF_FILENAME'] = filename
    os.environ['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    os.environ['PROCESSED_FOLDER'] = PROCESSED_FOLDER
    os.environ['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    
    command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output {notebook_path} --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.allow_errors=True"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"Notebook execution output: {output.strip()}")  # Additional debug output
            #emit_progress(output.strip())
    
    stderr = process.stderr.read()
    if stderr:
        #emit_progress(f"Notebook execution error: {stderr}")
        print(f"Notebook execution error: {stderr}")
    else:
        # Get the CSV data from the notebook output
        csv_data = process.stdout.read()
        
        # Emit the CSV data to the browser
        emit('csv_data', {'data': csv_data}, broadcast=True)

        # Only move the file if there's no error
        #filepath = os.path.join(UPLOAD_FOLDER, filename)
        #processed_filepath = os.path.join(PROCESSED_FOLDER, filename)
        #shutil.move(filepath, processed_filepath)
        #emit_progress(f"Moved {filename} to processed folder.")
        #print(f"Moved {filename} to processed folder.")
    #return redirect(url_for('index'))  # Redirect to the index page or any other appropriate page
    #return make_response("", 200)  # Return an empty response with a 200 OK status
    #return jsonify({'status': 'success', 'message': 'Notebook executed and file processed successfully'})
    emit_progress("resetting soon")
    socketio.emit('notebook_execution_complete', {'status': 'success', 'message': 'Notebook executed successfully'})
    #sleep(50)
    return render_template('index.html', filename=filename)
from openai import OpenAI
from openai import OpenAI   

@app.route('/query', methods=['POST'])
def query_model():
    client = OpenAI()
    user_query = request.form['user_query']
    system_message = "Marv is a factual chatbot that provides complete specifications based on user requirements."
    fine_tuned_model_id = os.environ.get('FINE_TUNED_MODEL_ID')
    if fine_tuned_model_id:
        set_current_model_id(fine_tuned_model_id)
    model_to_use = get_current_model_id()

    completion = client.chat.completions.create(
        model=model_to_use,  # Replace with your fine-tuned model ID
        
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Please provide the complete specification for the following requirements: {user_query}"}
        ]
    )

    message_content = completion.choices[0].message.content
    print("Model used:", model_to_use)
    print(f"User query: {user_query}")
    print(f"Marv's response: {message_content}")

    return message_content

@socketio.on('message_from_notebook')
def handle_message_from_notebook(message):
    print('Message from notebook:', message)
    #emit('message_to_browser', {'data': message})
    if message != 'csv_data':
        emit_progress(message)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, log_output=True, use_reloader=False)
