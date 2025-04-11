from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model

# Perform tracking with the model
results = model.track("test_1.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack