import subprocess
import time
import sys

def run_server():
    while True:
        print("Starting Flask server...")
        process = subprocess.Popen([sys.executable, 'app.py'])
        process.wait()
        print("Server stopped. Restarting in 5 seconds...")
        time.sleep(5)

if __name__ == '__main__':
    run_server()