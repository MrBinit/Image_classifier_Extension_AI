from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

save_directory = "/home/binit/image_classification/models"

processor.save_pretrained(save_directory)
model.save_pretrained(save_directory)