import os
import shutil

# 데이터 디렉토리 설정
RAW_DIR = "raw_data"              #  원본 이미지
ANNOTATED_DIR = "annotated_data"  # bbox 추출 후 저장
YOLO_DATASET_DIR = "yolo_dataset" # YOLO 학습용 데이터셋

def prepare_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_DIR, "labels", "train"), exist_ok=True)

def move_labeled_data():
    '''
    생성된 결과물 YOLO 학습 폴더로 이동
    '''

    image_src = os.path.join(ANNOTATED_DIR, "bbox_preview.jpg")
    label_src = os.path.join(ANNOTATED_DIR, "sample.txt")

    image_dst = os.path.join(YOLO_DATASET_DIR, "images", "train", "cat.jpg")
    label_dst = os.path.join(YOLO_DATASET_DIR, "labels", "train", "cat.txt")

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copyfile(image_src, image_dst)
        shutil.copyfile(label_src, label_dst)
        print("학습용 이미지/라벨 복사 완료")
    else:
        print("bbox 추출 결과가 없습니다.")