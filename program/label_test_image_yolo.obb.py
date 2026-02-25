import cv2
import os
import numpy as np

# --- CONFIGURATION ---
IMAGE_FOLDER = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\image_original"
LABEL_FOLDER = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\label_original"
OUTPUT_FOLDER = r"D:\semester 6\Project_Zero\picture\data_yolo_obb\original\test_label"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

CLASS_MAPPING = {
    0: "bus number"
}

COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0)
}

def denormalize_points(points, width, height):
    """Convert normalized OBB points to pixel coordinates"""
    pixel_points = []
    for i in range(0, len(points), 2):
        x = int(points[i] * width)
        y = int(points[i+1] * height)
        pixel_points.append([x, y])
    return np.array(pixel_points, dtype=np.int32)

def check_dataset():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg','.png','.jpeg'))]
    
    print(f"Found {len(image_files)} images")

    for img_file in image_files:
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        height, width, _ = image.shape

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(LABEL_FOLDER, label_file)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 9:
                    print(f"Invalid OBB label in {label_file}")
                    continue

                class_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))

                points = denormalize_points(coords, width, height)

                color = COLORS.get(class_id, (255,255,255))
                class_name = CLASS_MAPPING.get(class_id, "Unknown")

                # Draw polygon (OBB)
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

                # Put class text
                x, y = points[0]
                cv2.putText(image, class_name, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            cv2.putText(image, "NO LABEL", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        output_path = os.path.join(OUTPUT_FOLDER, img_file)
        cv2.imwrite(output_path, image)

    print("Done!")

if __name__ == "__main__":
    check_dataset()
