import subprocess

if __name__ == "__main__":
    try:
        print("Visualizing dataset effects...")
        subprocess.run(["python3", "visualize_data_effects.py"], check=True)

        print("Training CNN with validation...")
        subprocess.run(["python3", "train_cnn_with_validation.py"], check=True)

        print("Workflow completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error in pipeline: {e}")
