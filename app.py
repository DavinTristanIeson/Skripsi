import sys
from scripts.config import Ansi, VirtualEnvPath
import subprocess
import os

venv = VirtualEnvPath.detect_venv("Venv")
SERVER_SERVICE_PATH = "server_service.py"
TOPIC_SERVICE_PATH = "topic_service.py"

env = os.environ.copy()
env["APP"] = "1"

p_server = subprocess.Popen([venv.fastapi, "run", SERVER_SERVICE_PATH], stdout=sys.stdout, env=env)
p_topic = subprocess.Popen([venv.python, TOPIC_SERVICE_PATH], stdout=sys.stdout, env=env)

try:
  print(f"Starting the services now. It might take a while... {Ansi.Error}You can shut down the application by pressing Ctrl + C{Ansi.End}")
  while True:
    input()
except:
  p_server.terminate()
  p_topic.terminate()
