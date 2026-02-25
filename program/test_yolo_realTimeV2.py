import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล
model = YOLO(r'D:\semester 6\Project_Zero\yolo\runs\detect\train_v2_DN\weights\best.pt')

# 2. เปิดวิดีโอ
video_path = r'D:\semester 6\Project_Zero\picture\video\vdo3.mp4'
cap = cv2.VideoCapture(video_path)

# ดึงค่า FPS และขนาดดั้งเดิม
original_fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / original_fps) if original_fps > 0 else 1

# --- ตั้งค่าขนาดหน้าจอแสดงผลที่ต้องการ ---
display_width = 800 

print(f"กำลังเริ่มประมวลผล...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. รัน YOLO (ปรับ imgsz=320 เพื่อความเร็วสูงสุด)
    # ใช้ stream=True เพื่อประหยัด RAM
    results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False, stream=True)

    # ดึงผลลัพธ์มาวาดลงบนเฟรม
    annotated_frame = frame
    for result in results:
        annotated_frame = result.plot()

    # 4. การจัดการขนาดภาพ (Resize) เพื่อไม่ให้ล้นหน้าจอ
    h, w = annotated_frame.shape[:2]
    aspect_ratio = h / w
    display_height = int(display_width * aspect_ratio)
    resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

    # 5. แสดงผล
    cv2.imshow('YOLOv8 Real-time (Resized)', resized_frame)

    # ใช้ wait_time ที่คำนวณจาก FPS จริง เพื่อไม่ให้วิดีโอเล่นช้า/เร็วเกินไป
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()