import os
import subprocess
import sys

file = os.path.join(__file__)
parent = os.path.dirname(file)

def pip_install(*args):
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "install"] + list(args),
        stderr=subprocess.STDOUT,
        )
    for line in output.decode().split("\n"):
        if "Successfully installed" in line:
            print(line)

pip_install('-r', os.path.join(parent, 'requirements.txt'))