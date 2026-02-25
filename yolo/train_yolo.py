import os
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# โหลดโมเดลเริ่มต้น
model = YOLO("D:\semester 6\Project_Zero\yolo\yolov8n.pt")

# เทรน
model.train(
    data = r"D:\semester 6\Project_Zero\dataset_bus\data_bus.yml",
    epochs = 50,   # จำนวนรอบที่เทรน
    imgsz = 640,
    patience = 20
)
