import os
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

SLOW_FACTOR = 0.5
CONF_BALL = 0.20
SMOOTH_WIN = 5
DEFAULT_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR
POSE_MODEL_PATH = MODEL_DIR / "yolov8n-pose.pt"
DET_MODEL_PATH = MODEL_DIR / "yolov8x.pt"

def analyze_video_from_path(
    input_path: str,
    output_path: str,
    font_path: str = None,
    slow_factor: float = None
):
    """영상 분석 서비스 (구현 예정)"""
    pass