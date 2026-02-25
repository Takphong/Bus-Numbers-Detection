from ultralytics import YOLO
import os

# 1. โหลดโมเดล
model = YOLO(r'D:\semester 6\Project_Zero\yolo\runs\detect\train_v2_DN\weights\best.pt') 

# กำหนดโฟลเดอร์ปลายทางที่อยากให้ไฟล์ไปอยู่
output_path = r'D:\semester 6\Project_Zero\yolo\runs\test_yolo'

# 2. สั่ง Predict โดยระบุที่เก็บไฟล์
results = model.predict(
    source = r'D:\semester 6\Project_Zero\picture\original\original_img\img_228.png', 
    save = True, 
    conf = 0.5,
    project = output_path, # กำหนดโฟลเดอร์หลัก
    name = 'output',        # กำหนดชื่อโฟลเดอร์ย่อย (ไฟล์จะอยู่ใน output_path/output)
    exist_ok = True         # ถ้ามีโฟลเดอร์อยู่แล้วให้เซฟทับลงไปเลย ไม่ต้องสร้าง output2, output3
)