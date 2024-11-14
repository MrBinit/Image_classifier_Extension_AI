import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

# Load configuration
with open("/home/binit/image_classification/config.yaml", "r") as file:
    config = yaml.safe_load(file)

raw_data_path = config["paths"]["raw_data"]
split_data_path = config["paths"]["split_data"]
train_ratio = config["splits"]["train"]
test_ratio = config["splits"]["test"]
validation_ratio = config["splits"]["validation"]

# sum of splits equals 1.0
if not abs(train_ratio + test_ratio + validation_ratio - 1.0) < 1e-6:
    raise ValueError("The sum of train, test, and validation ratios must equal 1.0.")

# Ensure directories exist 
def make_dir_structure(base_path):
    for split in ["train", "validation", "test"]:
        for category in ["real", "AI"]:
            os.makedirs(os.path.join(base_path, split, category), exist_ok=True)

# clear existing files in split_data_path
def clear_split_data(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Cleared existing split data at {base_path}")

clear_split_data(split_data_path)
make_dir_structure(split_data_path)

valid_formats = (".jpeg", ".jpg", ".png")
real_files = [os.path.join(raw_data_path, "real", f)
              for f in os.listdir(os.path.join(raw_data_path, "real"))
              if f.lower().endswith(valid_formats)]
ai_files = [os.path.join(raw_data_path, "AI", f)
            for f in os.listdir(os.path.join(raw_data_path, "AI"))
            if f.lower().endswith(valid_formats)]

def split_and_save_data(files, base_path, category):
    train_files, temp_files = train_test_split(
        files, test_size=(validation_ratio + test_ratio), random_state=42
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio / (validation_ratio + test_ratio), random_state=42
    )

    def copy_files(file_list, split_name):
        for f in file_list:
            shutil.copy(f, os.path.join(base_path, split_name, category))

    copy_files(train_files, "train")
    copy_files(val_files, "validation")
    copy_files(test_files, "test")

    print(f"{category.capitalize()} files: {len(train_files)} for training, "
          f"{len(val_files)} for validation, {len(test_files)} for testing.")

split_and_save_data(real_files, split_data_path, "real")
split_and_save_data(ai_files, split_data_path, "AI")

print("Data has been split and saved successfully.")
