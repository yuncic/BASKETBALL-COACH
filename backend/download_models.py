#!/usr/bin/env python3
"""빌드 시점에 YOLO 모델을 미리 다운로드하는 스크립트"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR
POSE_MODEL_PATH = MODEL_DIR / "yolov8n-pose.pt"
DET_MODEL_PATH = MODEL_DIR / "yolov8x.pt"

print("📥 YOLO 모델 다운로드 시작...")

try:
    print(f"📥 Pose model 다운로드 중: {POSE_MODEL_PATH}")
    pose_model = YOLO("yolov8n-pose.pt")
    print(f"✅ Pose model 다운로드 완료")
except Exception as e:
    print(f"❌ Pose model 다운로드 실패: {e}")
    sys.exit(1)

try:
    print(f"📥 Detection model 다운로드 중: {DET_MODEL_PATH}")
    det_model = YOLO("yolov8x.pt")
    print(f"✅ Detection model 다운로드 완료")
except Exception as e:
    print(f"❌ Detection model 다운로드 실패: {e}")
    sys.exit(1)

print("✅ 모든 모델 다운로드 완료!")

