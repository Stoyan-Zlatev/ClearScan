
import subprocess

cmd_template_feature_extraction = ["./.venv/Scripts/python.exe","-m","src.ensemble.run_feature_extraction", "--k"]
cmd_template_random_forest = ["./.venv/Scripts/python.exe", "-m", "src.ensemble.run_train_random_forest", "--perform_k_fold", "False", "--k_fold_n", "10", "--kmeans_k"]

for i in range(27,36,1):
    try:
        print(f"Executing for k={i}")
        result = subprocess.run(cmd_template_random_forest + [str(i)], check=True, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error occured :(\n{e.stderr}")
        exit()