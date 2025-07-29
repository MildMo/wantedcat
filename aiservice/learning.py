from service import train_yolo_model
import os

DATA_CONFIG = "config.yaml"

if __name__ == "__main__":
    if not os.path.exists(DATA_CONFIG):
        raise FileNotFoundError(f"{DATA_CONFIG} 파일이 존재하지 않습니다.")

    results = train_yolo_model()