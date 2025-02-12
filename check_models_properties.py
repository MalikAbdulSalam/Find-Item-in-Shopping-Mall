from ultralytics import YOLO

# Load the model
model = YOLO('models/yolov8s.pt')  # replace with the path to your best.pt file if it's in a different location

# Display model information
print("Model Summary:")
model.info()  # displays a summary of the model, including architecture and parameter count

# Display model classes
print("\nClasses:")
print(model.names)  # prints a dictionary of class names

# Optionally, print additional model details
print("\nOther Model Details:")
# print(f"Model Path: {model.model_path}")
# print(f"Input Shape: {model.input_shape}")
print(f"Number of Classes: {len(model.names)}")
# print(f"Device: {model.device}")
