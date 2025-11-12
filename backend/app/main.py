from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import analyze

import os
from ultralytics import YOLO

def ensure_model(model_name, url):
    if not os.path.exists(model_name):
        print(f"📦 모델 {model_name} 이(가) 없습니다. 자동 다운로드 중...")
        os.system(f"wget -O {model_name} {url}")
        print("✅ 다운로드 완료!")

# YOLO 모델 자동 확보
ensure_model("yolov8x.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt")
ensure_model("yolov8n-pose.pt", "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")

pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8x.pt")

app = FastAPI(title="Shooting Analyzer API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Report-Path", "x-report-path", "X-Report-Base64", "x-report-base64"],
)

# 라우터 등록
app.include_router(analyze.router)


@app.get("/")
async def root():
    return {"message": "Shooting Analyzer API"}


@app.get("/status")
async def health_check():
    return {"status": "I'm Working!"}