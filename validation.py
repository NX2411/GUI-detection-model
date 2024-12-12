import time
from ultralytics import YOLO

if __name__ == '__main__':
    # YOLO 모델 로드
    model = YOLO("D:/Yolov8/Save_final/weights/best.pt")

    # 검증 데이터셋 추론 실행 및 시간 측정
    start_time = time.time()

    # 검증 데이터셋에 대한 추론 실행
    metrics = model.val()

    # 추론 종료 시간 기록
    end_time = time.time()

    # 총 추론 시간 계산
    total_time = end_time - start_time

    # 추론한 이미지 수 (검증 데이터셋 크기)
    num_images = metrics['total']  # 검증에 사용된 총 이미지 수

    # FPS 계산
    fps = num_images / total_time
    print(f"FPS: {fps:.2f}")

    # 클래스별 mAP 확인
    print("Per-class mAP:")
    print(metrics['per_class_map'])
