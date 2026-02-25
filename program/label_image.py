import cv2
import os

IMAGE_PATH = r"D:\semester 6\Project_Zero\picture\data_yolo_normal\original\original_img\img_334.png"

LABEL_DIR = r"D:\semester 6\Project_Zero\picture\data_yolo_normal\original\original_label"
CLASS_ID = 0

os.makedirs(LABEL_DIR, exist_ok=True)


img = cv2.imread(IMAGE_PATH)
h, w, _ = img.shape

bbox = cv2.selectROI("Select Bus Number", img, False, True)
cv2.destroyAllWindows()

x, y, bw, bh = bbox

x_center = (x + bw / 2) / w
y_center = (y + bh / 2) / h
width = bw / w
height = bh / h

image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
label_path = os.path.join(LABEL_DIR, image_name + ".txt")

with open(label_path, "w") as f:
    f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Done:", label_path)
