import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, DefaultDataCollator
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

# Load configuration from config.yaml
with open("/home/binit/image_classification/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load paths and hyperparameters from config
train_data_dir = config["paths"]["train_dir"]
val_data_dir = config["paths"]["val_dir"]
output_dir = config["paths"]["model_save_dir"]
log_dir = config["paths"]["log_dir"]
checkpoint_dir = "/home/binit/image_classification/checkpoints"

# Create checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

# Load model and hyperparameters
model_name = config["models"]["model_name"]
batch_size = int(config["hyperparameters"]["batch_size"])
learning_rate = float(config["hyperparameters"]["learning_rate"])
num_epochs = int(config["hyperparameters"]["num_epochs"])
weight_decay = float(config["hyperparameters"]["weight_decay"])

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)

# Load the pre-trained model
model = AutoModelForImageClassification.from_pretrained(model_name)

# Dataset Class
class PTImageDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224)):
        self.data_dir = data_dir
        self.files = []
        self.image_size = image_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".pt"):
                    self.files.append(os.path.join(root, filename))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = torch.load(file_path, weights_only=True)
        label = 0 if 'AI' in file_path else 1
        if image.dim() == 2:
            image = image.unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
        image = self.normalize(image.float() / 255.0)
        return {"pixel_values": image, "labels": label}

# Load datasets
train_dataset = PTImageDataset(train_data_dir)
val_dataset = PTImageDataset(val_data_dir)
data_collator = DefaultDataCollator()

# Set up TensorBoard writer
writer = SummaryWriter(log_dir)

# Define compute_metrics function for accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Define training arguments with epoch-only logging
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",  
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    logging_dir=log_dir,
    report_to="tensorboard",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Early stopping and checkpointing parameters
patience = 3
min_delta = 0.001
best_val_loss = float("inf")
epochs_no_improve = 0
best_models = []  # List to store paths to top 3 model checkpoints

# Training loop with early stopping and checkpoint saving
for epoch in range(num_epochs):
    # Train and evaluate
    train_output = trainer.train()
    eval_output = trainer.evaluate()

    # Extract metrics
    train_loss = train_output.training_loss
    val_loss = eval_output['eval_loss']
    train_accuracy = train_output.metrics.get('train_accuracy', None)
    val_accuracy = eval_output['eval_accuracy']

    # Print metrics for reference
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    if train_accuracy is not None and val_accuracy is not None:
        print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Write metrics to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    if train_accuracy is not None:
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    # Early stopping and checkpointing logic
    if val_loss < best_val_loss - min_delta:
        # Update best validation loss
        best_val_loss = val_loss
        epochs_no_improve = 0  # Reset patience counter

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.bin")
        model.save_pretrained(checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

        # Manage top 3 checkpoints
        best_models.append((val_loss, checkpoint_path))
        best_models.sort()  # Sort by val_loss, lowest first

        if len(best_models) > 3:
            # Remove the checkpoint with the highest validation loss if more than 3
            _, model_to_remove = best_models.pop()
            os.remove(model_to_remove)
            print(f"Removed checkpoint: {model_to_remove}")
    else:
        epochs_no_improve += 1

    # Early stopping trigger
    if epochs_no_improve >= patience:
        print("Early stopping triggered!")
        break

# Close the TensorBoard writer
writer.close()
