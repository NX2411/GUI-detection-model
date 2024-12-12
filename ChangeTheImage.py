import os
import cv2
import numpy as np

# Define the class names and their corresponding colors
class_names = [
    "BackgroundImage", "Bottom_Navigation", "Card", "CheckedTextView", "Drawer",
    "EditText", "Icon", "Image", "ImageButton", "List", "Modal", "Multi_Tab",
    "PageIndicator", "Switch", "Text", "TextButton", "TextImageButton", "Toolbar",
    "UpperTaskBar"
]

# Assign a color to each class (you can customize these colors)
class_colors = {
    "BackgroundImage": (255, 0, 0),
    "Bottom_Navigation": (0, 255, 0),
    "Card": (0, 0, 255),
    "CheckedTextView": (255, 255, 0),
    "Drawer": (255, 0, 255),
    "EditText": (0, 255, 255),
    "Icon": (128, 0, 0),
    "Image": (0, 128, 0),
    "ImageButton": (0, 0, 128),
    "List": (128, 128, 0),
    "Modal": (128, 0, 128),
    "Multi_Tab": (0, 128, 128),
    "PageIndicator": (64, 0, 0),
    "Switch": (0, 64, 0),
    "Text": (0, 0, 64),
    "TextButton": (64, 64, 0),
    "TextImageButton": (64, 0, 64),
    "Toolbar": (0, 64, 64),
    "UpperTaskBar": (192, 0, 0),
}

def process_folder(folder_path):
    images_path = os.path.join(folder_path, 'images')
    labels_path = os.path.join(folder_path, 'labels')
    output_path = os.path.join(folder_path, 'output_images')
    os.makedirs(output_path, exist_ok=True)

    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_path, image_file)
            label_path = os.path.join(labels_path, label_file)
            
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            height, width, _ = image.shape

             # 이미지를 하얀색으로 덮어씌움 (배경 처리)
            image[:] = (255, 255, 255)  # RGB 하얀색으로 이미지 초기화

            bounding_boxes = []

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cls_name = class_names[cls_id]
                    cx, cy, w, h = map(float, parts[1:5])

                    # Convert from YOLO format to bounding box coordinates
                    x1 = int((cx - w / 2) * width)
                    y1 = int((cy - h / 2) * height)
                    x2 = int((cx + w / 2) * width)
                    y2 = int((cy + h / 2) * height)

                    color = class_colors.get(cls_name, (255, 255, 255))
                    area = (x2 - x1) * (y2 - y1)
                    bounding_boxes.append((x1, y1, x2, y2, color, area))

            # 면적을 기준으로 큰 순서로 정렬
            bounding_boxes.sort(key=lambda box: box[5], reverse=True)

            # 큰 바운딩 박스부터 순서대로 그림
            for (x1, y1, x2, y2, color, _) in bounding_boxes:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
                
            output_image_path = os.path.join(output_path, image_file)
            cv2.imwrite(output_image_path, image)
            print(f"Processed and saved: {output_image_path}")

# Process the train, test, and valid folders
base_folder = 'Yolov8_UI-similarity_1331-2'
for sub_folder in ['train', 'valid', 'test']:
    process_folder(os.path.join(base_folder, sub_folder))
