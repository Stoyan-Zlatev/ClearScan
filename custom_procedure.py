
import subprocess

cmd_template = ["./.venv/Scripts/python.exe","-m","src.ensemble.run_feature_extraction", "--k"]

for i in range(18,36,1):
    try:
        print(f"Executing for k={i}")
        result = subprocess.run(cmd_template + [str(i)], check=True, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error occured :(\n{e.stderr}")
        exit()