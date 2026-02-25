import cv2
import os

# ====== CONFIG ======
IMAGE_PATH = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\image_original\img_4.png"
LABEL_DIR = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\label_original"
CLASS_ID = 0
# ====================

points = []

# Create label folder if not exists
os.makedirs(LABEL_DIR, exist_ok=True)

def click_event(event, x, y, flags, param):
    global points, img

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Draw point
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # Draw line between points
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)

        cv2.imshow("Image", img)

        print(f"Point {len(points)}: ({x}, {y})")

        if len(points) == 4:
            # Close polygon
            cv2.line(img, points[3], points[0], (255, 0, 0), 2)
            cv2.imshow("Image", img)
            save_label()

def save_label():
    global points, img_w, img_h

    label_line = str(CLASS_ID)

    for x, y in points:
        x_norm = x / img_w
        y_norm = y / img_h
        label_line += f" {x_norm:.6f} {y_norm:.6f}"

    # Create label filename
    image_name = os.path.basename(IMAGE_PATH)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, txt_name)

    with open(label_path, "w") as f:
        f.write(label_line)

    print("\n Label Saved at:")
    print(label_path)
    print(label_line)

# Load image
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Cannot load image. Check path.")
    exit()

img_h, img_w = img.shape[:2]

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("Click 4 points (clockwise).")
cv2.waitKey(0)
cv2.destroyAllWindows()
