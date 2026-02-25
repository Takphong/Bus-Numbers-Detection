import cv2
import os

# ====== CONFIG ======
IMAGE_PATH = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\image_original\img_18.png"
LABEL_DIR = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\label_original"
CLASS_ID = 0
# ====================

points = []

# Create label folder if not exists
os.makedirs(LABEL_DIR, exist_ok=True)

# Load image
original_img = cv2.imread(IMAGE_PATH)q

if original_img is None:
    print("Cannot load image. Check path.")
    exit()

img_h, img_w = original_img.shape[:2]
img = original_img.copy()


def redraw():
    global img
    img = original_img.copy()

    # Draw points and lines
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        if i > 0:
            cv2.line(img, points[i - 1], points[i], (255, 0, 0), 2)

    # Close polygon if 4 points
    if len(points) == 4:
        cv2.line(img, points[3], points[0], (255, 0, 0), 2)

    cv2.imshow("Image", img)


def click_event(event, x, y, flags, param):
    global points

    # Left click → Add point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            redraw()

    # Right click → Remove last point
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            removed = points.pop()
            print(f"Removed: {removed}")
            redraw()


def save_label():
    global points

    if len(points) != 4:
        print("Need exactly 4 points to save!")
        return

    label_line = str(CLASS_ID)

    for x, y in points:
        x_norm = x / img_w
        y_norm = y / img_h
        label_line += f" {x_norm:.6f} {y_norm:.6f}"

    image_name = os.path.basename(IMAGE_PATH)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, txt_name)

    with open(label_path, "w") as f:
        f.write(label_line)

    print("\nLabel Saved at:")
    print(label_path)
    print(label_line)


cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

print("Left Click = Add point")
print("Right Click = Remove last point")
print("Press 'r' = Reset")
print("Press 's' = Save")
print("Press 'q' = Quit")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        points = []
        redraw()
        print("Reset!")

    elif key == ord('s'):
        save_label()

    elif key == ord('q'):
        break

cv2.destroyAllWindows()