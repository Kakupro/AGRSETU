
import os
import shutil
import subprocess
import time
import stat

# --- Configuration ---
DATASET_DIR = "dataset"
TEMP_DIR = "temp_repo_grains"

# Repositories
WHEAT_REPO = "https://github.com/deepakrana47/Wheat-quality-detector-2.git"
RICE_REPO = "https://github.com/NikhilKP631197/Count-Total-Rice-Grains-Total-Broken-Rice-Grains.git"

def on_rm_error(func, path, exc_info):
    # Handle read-only files on Windows
    os.chmod(path, stat.S_IWRITE)
    try:
        os.unlink(path)
    except:
        pass

def run_command(cmd, cwd=None):
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing {cmd}: {e}")
        return False

def clean_temp():
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR, onerror=on_rm_error)
        except Exception as e:
            print(f"Warning: Could not fully clean temp dir: {e}")

def setup_dataset():
    # 1. Prepare Dataset Folder (Don't delete if it already has data, just add/overwrite)
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    clean_temp()

    # 2. Download Wheat Data
    print(f"\n--- Downloading Wheat Quality Data ---")
    if run_command(f"git clone --depth 1 {WHEAT_REPO} {TEMP_DIR}"):
        time.sleep(1) # Wait for file handles to release
        
        # Search strategy: Look for 'healthy' and 'damaged' across the repo
        found_wheat = 0
        for root, dirs, files in os.walk(TEMP_DIR):
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 5:
                # Identify class by folder name
                folder_name = os.path.basename(root).lower()
                dest_sub = None
                
                if "health" in folder_name or "1" == folder_name:
                    dest_sub = "Wheat_Healthy"
                elif "amag" in folder_name or "bad" in folder_name or "0" == folder_name:
                    dest_sub = "Wheat_Damaged"
                
                if dest_sub:
                    full_dest = os.path.join(DATASET_DIR, dest_sub)
                    os.makedirs(full_dest, exist_ok=True)
                    for img in images:
                        try:
                            shutil.copy(os.path.join(root, img), full_dest)
                        except: pass
                    print(f"  -> Copied {len(images)} images to {dest_sub}")
                    found_wheat += len(images)
        
        if found_wheat == 0:
            print("Warning: Could not automatically find labeled Wheat images.")
    
    clean_temp()

    # 3. Download Rice Data
    print(f"\n--- Downloading Rice Quality Data ---")
    if run_command(f"git clone --depth 1 {RICE_REPO} {TEMP_DIR}"):
        time.sleep(1)
        
        found_rice = 0
        for root, dirs, files in os.walk(TEMP_DIR):
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) > 5:
                folder_name = os.path.basename(root).lower()
                dest_sub = None
                
                if "full" in folder_name or "proper" in folder_name:
                    dest_sub = "Rice_Full"
                elif "broken" in folder_name:
                    dest_sub = "Rice_Broken"
                
                if dest_sub:
                    full_dest = os.path.join(DATASET_DIR, dest_sub)
                    os.makedirs(full_dest, exist_ok=True)
                    for img in images:
                        try:
                            shutil.copy(os.path.join(root, img), full_dest)
                        except: pass
                    print(f"  -> Copied {len(images)} images to {dest_sub}")
                    found_rice += len(images)

    clean_temp()

    # Validation
    print("\n--- Summary ---")
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Dataset is ready in '{DATASET_DIR}/'.")
    print(f"Classes found: {classes}")
    print("You can now run 'python train.py'.")

if __name__ == "__main__":
    setup_dataset()
