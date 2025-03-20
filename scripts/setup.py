import os
import sys

sys.path.append(os.getcwd())

import subprocess
import argparse
import scripts.config
from scripts.config import Ansi


parser = argparse.ArgumentParser(
  prog="Setup Project",
  description="This script is used to setup the virtual environment and its dependencies. Your global environment will not be polluted by the dependencies of this project; all of the library versions are self-contained in the Venv directory.",
)
parser.add_argument("--dev", dest="is_dev", action="store_const", default=False, const=True, help="Installs the dependencies into the virtual environment. The dependencies in this file do not have their versions pinned; instead, the versions are resolved by pip-compile. This also means that there's no guarantee the versions produced by two pip-compile calls will be the same. Do not use unless you're a developer and want to change the dependencies.")
args = parser.parse_args()

if not os.path.exists(scripts.config.VIRTUALENV_NAME):
  scripts.config.VirtualEnvPath.create_venv(scripts.config.VIRTUALENV_NAME)

venvpaths = scripts.config.VirtualEnvPath.detect_venv(scripts.config.VIRTUALENV_NAME)
subprocess.run(venvpaths.activate_venv, check=True)

if not os.path.exists(venvpaths.pip_compile) or not os.path.exists(venvpaths.pip_sync):
  print(f"{Ansi.Warning}Found no existing pip-tools installation. Installing them from PyPy...{Ansi.End}")
  subprocess.run([venvpaths.pip, "install", "pip-tools"], check=True)
if args.is_dev:
  print(f"{Ansi.Grey}Running pip-compile to get the new requirements.lock file.{Ansi.End}")
  subprocess.run([venvpaths.pip_compile, scripts.config.REQUIREMENTS_PATH, "-o", scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"], check=True)
print(f"{Ansi.Grey}Syncing requirements.lock file with the virtual environment.{Ansi.End}")
subprocess.run([venvpaths.pip_sync, scripts.config.REQUIREMENTS_LOCK_PATH, "--verbose"], check=True)

print(f"{Ansi.Grey}Downloading spacy pipeline model...{Ansi.End}")
subprocess.run([venvpaths.python, "-m", "spacy", "download", "en_core_web_sm"])

print(f"{Ansi.Success}The virtual environment has been set-up successfully. You may now run \"python app.py\" to start the app.{Ansi.End}")