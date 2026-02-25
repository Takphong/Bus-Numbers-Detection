import os
import cv2

# ===== PATHS =====
input_dir = r"D:\semester 6\Project_Zero\picture\data_yolo_normal\original\original_img"
output_dir = r"D:\semester 6\Project_Zero\picture\data_yolo_normal\original\original_img"

os.makedirs(output_dir, exist_ok=True)

# ===== PROCESS (Rename Only + Convert to PNG) =====
img_number = 334  # เริ่มนับเลข

for name in os.listdir(input_dir):
    # ตรวจสอบว่าเป็นไฟล์ภาพเท่านั้น
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, name)
    img = cv2.imread(img_path)

    # ถ้าอ่านรูปไม่ได้ ให้ข้าม
    if img is None:
        continue

    # ตั้งชื่อใหม่เป็น png เสมอ
    new_name = f"img_{img_number}.png"
    save_path = os.path.join(output_dir, new_name)

    cv2.imwrite(save_path, img)

    print(f"Saved: {new_name}")
    img_number += 1

print("Renaming and PNG conversion finished")