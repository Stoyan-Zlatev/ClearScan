import kagglehub
import os
import shutil

from src.common.path_utils import resolve_path

print("Downloading dataset...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

print("Copying dataset...")

# Define source directories and destination directories
source_dirs = ['test', 'train', 'val']
dest_dir = '/data/raw'
categories = ['NORMAL', 'PNEUMONIA']

dest_path_root = resolve_path(dest_dir)
source_path_root = os.path.join(path, "chest_xray")

# Create destination folders
os.makedirs(dest_path_root, exist_ok=True)
for category in categories:
    os.makedirs(os.path.join(dest_path_root, category), exist_ok=True)

# Copy files
for source in source_dirs:
    for category in categories:
        source_path = os.path.join(os.path.join(source_path_root, source), category)
        dest_path = os.path.join(dest_path_root, category)
        
        if os.path.exists(source_path):
            for file_name in os.listdir(source_path):
                source_file = os.path.join(source_path, file_name)
                dest_file = os.path.join(dest_path, file_name)
                
                if os.path.isfile(source_file):  # Ensure it's a file
                    shutil.copy2(source_file, dest_file)

print(f"Dataset copied successfully to {dest_dir}")