import os
import cv2
import numpy as np
from colorsys import hsv_to_rgb

# Define the class names
class_names = [
    "BackgroundImage", "Bottom_Navigation", "Card", "CheckedTextView", "Drawer",
    "EditText", "Icon", "Image", "ImageButton", "List", "Modal", "Multi_Tab",
    "PageIndicator", "Switch", "Text", "TextButton", "TextImageButton", "Toolbar",
    "UpperTaskBar"
]

def generate_distinct_colors(n_colors):
    colors = {}
    for i, name in enumerate(class_names):
        # HSV 색상환에서 골고루 색상을 선택
        hue = i / n_colors
        # 채도와 명도를 높게 설정하여 구분성 증가
        saturation = 0.9
        value = 0.9
        
        # HSV to RGB 변환
        rgb = hsv_to_rgb(hue, saturation, value)
        # RGB to BGR (OpenCV format) 및 0-255 스케일로 변환
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors[name] = bgr
    
    return colors

# 구분되는 색상 생성
class_colors = generate_distinct_colors(len(class_names))

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

            # 원본 이미지 크기 가져오기
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # 흰색 배경 생성 (RGB)
            image = np.ones((height, width, 3), dtype=np.uint8) * 255

            # 바운딩 박스 정보 수집
            bounding_boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cls_name = class_names[cls_id]
                    cx, cy, w, h = map(float, parts[1:5])

                    # YOLO 좌표를 픽셀 좌표로 변환
                    x1 = int((cx - w/2) * width)
                    y1 = int((cy - h/2) * height)
                    x2 = int((cx + w/2) * width)
                    y2 = int((cy + h/2) * height)

                    color = class_colors[cls_name]
                    area = (x2 - x1) * (y2 - y1)
                    bounding_boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'color': color,
                        'area': area
                    })

            # 면적 기준 내림차순 정렬 (큰 컴포넌트가 먼저 그려지도록)
            bounding_boxes.sort(key=lambda x: x['area'], reverse=True)

            # 바운딩 박스 그리기 (경계선 없이 색상으로만 구분)
            for box in bounding_boxes:
                x1, y1, x2, y2 = box['coords']
                color = box['color']
                # 채워진 사각형만 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

            # 이미지 저장
            output_image_path = os.path.join(output_path, image_file)
            cv2.imwrite(output_image_path, image)
            print(f"Processed and saved: {output_image_path}")

# 폴더 처리 실행
base_folder = 'Yolov8_UI-similarity_ver4-1'
for sub_folder in ['train', 'valid', 'test']:
    folder_path = os.path.join(base_folder, sub_folder)
    print(f"Processing {sub_folder} folder...")
    process_folder(folder_path)
    print(f"Completed processing {sub_folder} folder")