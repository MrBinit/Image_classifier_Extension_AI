import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
from torchvision import transforms
import numpy as np

# Load configuration from config.yaml
with open("/home/binit/image_classification/config.yaml", "r") as file:
    config = yaml.safe_load(file)

train_data_dir = config["paths"]["train_dir"]
val_data_dir = config["paths"]["val_dir"]
output_dir = config["paths"]["model_save_dir"]
log_dir = config["paths"]["log_dir"]

model_name = config["models"]["model_name"]

# Cast hyperparameters to the correct types
batch_size = int(config["hyperparameters"]["batch_size"])
learning_rate = float(config["hyperparameters"]["learning_rate"])
num_epochs = int(config["hyperparameters"]["num_epochs"])
weight_decay = float(config["hyperparameters"]["weight_decay"])
dropout = float(config["hyperparameters"].get("dropout", 0.3))  # Default to 0.3 if not specified
save_steps = int(config["hyperparameters"]["save_steps"])
max_checkpoints = int(config["hyperparameters"].get("max_checkpoints", 3))

# Load the pre-trained model
model = AutoModelForImageClassification.from_pretrained(model_name)

# Custom Dataset Class
class PTImageDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224)):
        self.data_dir = data_dir
        self.files = []
        self.image_size = image_size  # Target image size
        
        # Define normalization transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Walk through the directory and gather all .pt files
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".pt"):
                    self.files.append(os.path.join(root, filename))
        
        print(f"Loaded {len(self.files)} files from {data_dir}")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = torch.load(file_path)  # Load the tensor directly

        # Infer label based on folder name
        label = 0 if 'AI' in file_path else 1

        # Ensure the image tensor has 3 dimensions (channels, height, width)
        if image.dim() == 2:  # If the image is a 2D tensor (grayscale)
            image = image.unsqueeze(0)  # Add a channel dimension for grayscale image
        elif image.dim() == 3 and image.size(0) not in [1, 3]:  # Single-channel image with wrong shape
            image = image.permute(2, 0, 1)  # Rearrange to (channels, height, width)

        # Convert to float and scale to [0, 1]
        image = image.float() / 255.0

        # Resize the image to the target size
        image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
        
        # Apply normalization
        image = self.normalize(image)

        return {"pixel_values": image, "labels": label}

# Load datasets
train_dataset = PTImageDataset(train_data_dir)
val_dataset = PTImageDataset(val_data_dir)

# Define data collator
data_collator = DefaultDataCollator()

# Define training arguments using config.yaml parameters
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    logging_dir=log_dir,
    save_total_limit=config["hyperparameters"].get("max_checkpoints", 3),  # Defaults to 3 if not specified
)

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Ensure logits and labels are tensors
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)
