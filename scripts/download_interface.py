import os
import sys
sys.path.append(os.getcwd())

import urllib.request
import zipfile
import shutil
from config import Ansi

INTERFACE_RELEASE_URL = "https://github.com/DavinTristanIeson/Skripsi-Frontend/releases/latest/download/interface.zip"
INTERFACE_DESTINATION = "views"
INTERFACE_DESTINATION_TEMP = "views/temp.zip"

if not os.path.exists(INTERFACE_DESTINATION):
  print(f"{Ansi.Warning}Creating {INTERFACE_DESTINATION} since it hasn't existed before.{Ansi.End}")
  os.mkdir(INTERFACE_DESTINATION)
else:
  shutil.rmtree(INTERFACE_DESTINATION)
  os.mkdir(INTERFACE_DESTINATION)
  

try:
  print(f"{Ansi.Grey}Fetching files from {INTERFACE_RELEASE_URL}...{Ansi.End}")
  urllib.request.urlretrieve(INTERFACE_RELEASE_URL, INTERFACE_DESTINATION_TEMP)
except Exception as e:
  print(f"{Ansi.Error}Failed to retrieve interface files from {INTERFACE_RELEASE_URL}. Error => {e}{Ansi.End}")
  exit(1)

print(f"{Ansi.Grey}Extracting files from {INTERFACE_DESTINATION_TEMP}...{Ansi.End}")
with zipfile.ZipFile(INTERFACE_DESTINATION_TEMP) as f:
  f.extractall(INTERFACE_DESTINATION)

if os.path.exists(INTERFACE_DESTINATION_TEMP):
  os.remove(INTERFACE_DESTINATION_TEMP)

print(f"{Ansi.Success}Interface files has been successfully downloaded to {INTERFACE_DESTINATION}. You should be able to access the interface through http://localhost:8000 now (restart the app first).{Ansi.End}")

