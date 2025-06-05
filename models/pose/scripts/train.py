from ultralytics import YOLO
import yaml
import os
# Load the model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "../data", "formatted","keypoints").format()
DATAYAML_PATH = os.path.join(SCRIPT_DIR, "../data", "formatted","keypoints", "data.yaml").format()

data_config = {
    "path": DATASET_DIR,
    "train": "images/train",
    "val": "images/val",
    "kpt_shape": [22, 3],
    "names": ["horse"],
    "keypoints": [
        "Nose", "Eye", "Nearknee", "Nearfrontfetlock", "Nearfrontfoot",
        "Offknee", "Offfrontfetlock", "Offfrontfoot", "Shoulder", "Midshoulder",
        "Elbow", "Girth", "Wither", "Nearhindhock", "Nearhindfetlock", "Nearhindfoot",
        "Hip", "Stifle", "Offhindhock", "Offhindfetlock", "Offhindfoot", "Ischium",
    ],
}

with open(DATAYAML_PATH, "w") as f:
    yaml.dump(data_config, f)


model = YOLO("yolov8n-pose.pt")  # or "yolov8n-pose.yaml" for training from scratch
# Train the model
model.train(
    task="pose",
    data=DATAYAML_PATH,
    epochs=100,
    imgsz=640,
    batch=128,
    project=os.path.join(SCRIPT_DIR, "../models")
)