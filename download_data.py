
import os
import subprocess
import shutil

# Configuration
REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
TARGET_DIR = "dataset"
TEMP_DIR = "temp_repo"
CLASSES_TO_DOWNLOAD = [
    "raw/color/Potato___Early_blight",
    "raw/color/Potato___Late_blight",
    "raw/color/Potato___Healthy",
    "raw/color/Tomato___Bacterial_spot",
    "raw/color/Tomato___Early_blight",
    "raw/color/Tomato___Healthy"
]

def run_command(command, cwd=None):
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")
        return False
    return True

def download_dataset():
    if os.path.exists(TARGET_DIR):
        print(f"'{TARGET_DIR}' already exists. Skipping download to avoid valid data overwrite.")
        print("Delete the 'dataset' folder if you want to re-download.")
        return

    print("Checking for Git...")
    if not run_command("git --version"):
        print("Git is not installed or not in PATH. Please install Git to download dataset automatically.")
        return

    # 1. Initialize empty git repo
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    print(f"Initializing repo in {TEMP_DIR}...")
    run_command("git init", cwd=TEMP_DIR)
    run_command(f"git remote add -f origin {REPO_URL}", cwd=TEMP_DIR)

    # 2. Configure Sparse Checkout
    print("Configuring sparse checkout...")
    run_command("git config core.sparseCheckout true", cwd=TEMP_DIR)
    
    sparse_path = os.path.join(TEMP_DIR, ".git", "info", "sparse-checkout")
    with open(sparse_path, "w") as f:
        for cls in CLASSES_TO_DOWNLOAD:
            f.write(f"{cls}\n")

    # 3. Pull Data (Master branch)
    print("Downloading specific classes (this may take a minute)...")
    if not run_command("git pull origin master", cwd=TEMP_DIR):
        print("Failed to pull data.")
        return

    # 4. Move Data to Target Directory
    print("Moving data to 'dataset' folder...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    source_base = os.path.join(TEMP_DIR, "raw", "color")
    
    if os.path.exists(source_base):
        for item in os.listdir(source_base):
            s = os.path.join(source_base, item)
            d = os.path.join(TARGET_DIR, item)
            if os.path.isdir(s):
                shutil.move(s, d)
                print(f"Moved {item}")
    else:
        print("Could not find downloaded data. The repository structure might have changed.")

    # 5. Cleanup
    print("Cleaning up...")
    # robust cleanup for windows
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Warning: Could not fully remove temp dir: {e}")

    print("\nDataset setup complete!")
    print(f"Images are now in '{TARGET_DIR}/'")

if __name__ == "__main__":
    download_dataset()
