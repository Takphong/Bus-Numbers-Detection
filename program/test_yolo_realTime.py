import cv2
from ultralytics import YOLO

# 1. โหลดโมเดลตัวที่เทรนมา
model = YOLO(r'D:\semester 6\Project_Zero\yolo\runs\detect\train_v2_DN\weights\best.pt')

# 2. เปิดวิดีโอ (หรือใส่ 0 หากต้องการใช้กล้อง Webcam)
video_path = r'D:\semester 6\Project_Zero\picture\video\vdo2.mp4'
cap = cv2.VideoCapture(video_path)

# เช็คความกว้าง/สูงของวิดีโอเพื่อความถูกต้องในการแสดงผล
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"กำลังเริ่มประมวลผลวิดีโอขนาด: {frame_width}x{frame_height}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. รัน YOLO แบบประหยัดทรัพยากร
    # - verbose=False: ปิดการ Print Log ใน Terminal เพื่อให้ลื่นขึ้น
    # - stream=True: ใช้ generator ในการจัดการเฟรม
    results = model.predict(source=frame, conf=0.5, verbose=False, stream=True)

    for result in results:
        # ดึงภาพที่วาด Box และ Label เรียบร้อยแล้วจาก YOLO โดยตรง
        annotated_frame = result.plot()

    # 4. แสดงผล
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # กด 'q' เพื่อหยุดรัน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()