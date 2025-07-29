from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

from service import extract_bboxes, train_yolo_model

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "캣밥바라기 AI API 서버입니다."}


@app.post("/bbox")
async def get_bboxes(file: UploadFile = File(...)):
    try:
        # 업로드된 이미지 저장
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # bbox 추출
        bboxes = extract_bboxes(image_path)

        # 응답 형식 정리
        bbox_list = [
            {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}
            for b in bboxes
        ]

        return JSONResponse(content={"filename": file.filename, "bboxes": bbox_list})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/train")
def trigger_training():
    try:
        results = train_yolo_model()
        return {"status": "✅ 학습 완료", "model_info": str(results)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)