import sys
from scripts.config import VirtualEnvPath
import subprocess

venv = VirtualEnvPath.detect_venv("Venv")
SERVER_SERVICE_PATH = "server_service.py"
TOPIC_SERVICE_PATH = "topic_service.py"

p_server = subprocess.Popen([venv.fastapi, "run", SERVER_SERVICE_PATH], stdout=sys.stdout)
p_topic = subprocess.Popen([venv.python, TOPIC_SERVICE_PATH], stdout=sys.stdout)

try:
  while True:
    input()
except:
  p_server.terminate()
  p_topic.terminate()
