import subprocess

def run_script(script_name):
    """
    Executes a Python script and logs the output.
    """
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(["python3", script_name], check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:\n{e.stderr}")
        raise

if __name__ == "__main__":
    try:
        # Step 1: Run preprocessing script
        run_script("run_preprocessing.py")

        # Step 2: Run feature extraction script
        run_script("run_feature_extraction.py")

        print("Pipeline executed successfully!")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
