import cv2
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog


TRAIN_DIR = "train"
IMAGE_PATH = None # absolute image path to predict
CONFIDENCE = 0.25

if not IMAGE_PATH:
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    IMAGE_PATH = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Load your trained model (adjust the path as needed)
model = YOLO(os.path.join(SCRIPT_DIR, "../models",TRAIN_DIR, "weights", "best.pt"))

# Load the image
image = cv2.imread(IMAGE_PATH)

# Run inference
results = model(image, show=False, conf=CONFIDENCE)  # conf can be adjusted

# Visualize results
for result in results:
    annotated_frame = result.plot()

# Show the image with keypoints
cv2.imshow("YOLOv8 Pose Inference", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()