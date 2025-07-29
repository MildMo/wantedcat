from service import extract_bboxes
import os

IMAGE_PATH = "sample.jpg"  # 업로드 이미지 경로

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"{IMAGE_PATH} 파일이 존재하지 않습니다.")

    bboxes = extract_bboxes(IMAGE_PATH)