from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

# ---------------------------
# 설정
# ---------------------------
MODEL_FILENAME = "yolov11m.pt"
REPO_ID = "Ultralytics/YOLO11"
OUTPUT_DIR = "outputs"
SAVE_LABEL_TXT = True


# ---------------------------
# BBox 추출 함수
# ---------------------------
def extract_bboxes(image_path: str, class_id: int = 0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 모델 로드
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    model = YOLO(model_path)

    # 2. 추론 수행
    results = model(image_path)

    # 3. bbox 좌표 (xyxy 포맷)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # 4. 시각화 이미지 저장
    preview_path = os.path.join(OUTPUT_DIR, "bbox_preview.jpg")
    results[0].save(filename=preview_path)
    print(f"시각화 저장: {preview_path}")

    # 5. YOLO 포맷 라벨 파일 저장 (class_id는 추후 class_name 매핑으로 교체 가능)
    if SAVE_LABEL_TXT:
        label_path = os.path.join(OUTPUT_DIR, "sample.txt")
        with open(label_path, "w") as f:
            for box in results[0].boxes.xywhn.cpu().numpy():
                f.write(f"{class_id} {box[0]} {box[1]} {box[2]} {box[3]}\n")
        print(f"라벨 저장: {label_path}")

    return boxes


# ---------------------------
# YOLO 모델 학습 함수
# ---------------------------
def train_yolo_model():
    DATA_CONFIG = "config.yaml"
    EPOCHS = 50
    PROJECT = "wantedcat_train"
    NAME = "cat_detector_v1"

    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError(f"{DATA_CONFIG} 파일이 존재하지 않습니다.")

    # 1. 모델 로드
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    model = YOLO(model_path)

    # 2. 학습 수행
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=640,
        project=PROJECT,
        name=NAME,
        pretrained=True
    )

    print(f"모델 학습 완료!\n 결과: runs/{PROJECT}/{NAME}/")
    return results